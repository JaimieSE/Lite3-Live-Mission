#!/usr/bin/env python3
"""Orbbec RGB-D mission perception with optional wide-camera search."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import mkdtemp
import threading
import time

import cv2
import torch
import numpy as np
import zxingcpp
from ultralytics import YOLO
from periodic_vlm import Camera
from yellow_signs import detect_yellow_triangles, SignTracker, crop_sign, fingerprint
from sign_vlm import identify
from competition import Competition, prepare_navigation, read_json, validate_profile
from live_store import LiveStore
from wide_handoff import WideHandoff

ROOT=Path(__file__).resolve().parent


def detect_qr(frame):
    codes = []
    text, points, _ = cv2.QRCodeDetector().detectAndDecode(frame)
    if points is not None:
        codes.append(dict(text=text, decoded=bool(text), source="OpenCV",
                          corners=points.reshape(4, 2).tolist()))
    for barcode in zxingcpp.read_barcodes(frame, formats=zxingcpp.BarcodeFormat.QRCode):
        position = barcode.position
        corners = [[point.x, point.y] for point in
                   (position.top_left, position.top_right, position.bottom_right, position.bottom_left)]
        center = np.mean(corners, axis=0)
        span = max(np.ptp(corners, axis=0))
        existing = next((qr for qr in codes if (not qr["decoded"] or qr["text"] == barcode.text)
                         and np.linalg.norm(np.mean(qr["corners"], axis=0) - center)
                         <= max(5, span * 0.2)), None)
        if existing is not None:
            if existing["decoded"]:
                continue
            codes.remove(existing)
        codes.append(dict(text=barcode.text, decoded=True, source="ZXing", corners=corners))
    return codes


def room_preview(wide_sample, rgbd_frame, now):
    canvas = np.zeros((440, 960, 3), dtype=np.uint8)

    def place(image, x, width):
        scale = min(width / image.shape[1], 405 / image.shape[0])
        size = tuple(max(1, round(length * scale)) for length in image.shape[1::-1])
        left = x + (width - size[0]) // 2
        canvas[30:30 + size[1], left:left + size[0]] = cv2.resize(image, size)

    room_label = 'Wide camera - waiting for a live image'
    if wide_sample is not None and now - wide_sample[0] <= 1:
        place(wide_sample[2], 0, 720)
        room_label = 'Wide camera - room view'
    if rgbd_frame is not None:
        place(rgbd_frame, 720, 240)
    cv2.putText(canvas, room_label, (12, 21), 0, .55, (255, 255, 255), 1)
    cv2.putText(canvas, 'RGB-D evidence' if rgbd_frame is not None else 'RGB-D idle / waiting',
                (727, 21), 0, .49, (255, 255, 255), 1)
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(ROOT.parent/'scene_understanding'/'yolo26n.pt'))
    p.add_argument('--confidence',type=float,default=.4)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--scene-understanding', action='store_true',
                   help='Classify RGB-D evidence for local v6.3 scoring via Claude')
    p.add_argument('--vlm',action='store_true',help='Enable paid Claude sign identification')
    p.add_argument('--dry-run',action='store_true',help='Save sign triggers locally; no API calls')
    p.add_argument('--vlm-model',default=os.environ.get('CLAUDE_MODEL','claude-opus-5-5'))
    p.add_argument('--max-requests',type=int,default=3)
    p.add_argument('--cooldown',type=float,default=30)
    p.add_argument('--min-area',type=int,default=300)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seconds',type=float,default=0)
    database = p.add_mutually_exclusive_group()
    database.add_argument('--db', type=Path, help='Write this run to a new database at this path')
    database.add_argument('--resume', type=Path, help='Continue an existing database intentionally')
    camera_source = p.add_mutually_exclusive_group()
    camera_source.add_argument('--rgbd', dest='camera_mode', action='store_const', const='rgbd',
                               help='Use the Orbbec RGB-D camera (default)')
    camera_source.add_argument('--wide', dest='camera_mode', action='store_const', const='wide',
                               help='Use only the built-in wide camera, including for detection')
    camera_source.add_argument('--wide-search', dest='camera_mode', action='store_const', const='wide_search',
                               help='Search with the wide camera and use the Orbbec to locate matched targets')
    p.set_defaults(camera_mode='rgbd')
    p.add_argument('--scout-fps', type=float, default=4.0, help='Maximum wide-search inference rate')
    p.add_argument('--locate-wide-qr', action='store_true',
                   help='Also request depth for QR codes already decoded on the wide camera')
    p.add_argument('--publish-mission-events', action='store_true',
                   help='Publish saved RGB-D evidence events to a separate running mission node')
    p.add_argument('--reid', action='store_true', help='Use BoxMOT and remember person appearances during this run')
    p.add_argument('--objectives', type=Path, help='JSON array of objectives for this attempt')
    p.add_argument('--competition', type=Path, help='Load a scoring profile and start a fresh attempt')
    p.add_argument('--navigation-state', type=Path,
                   default=Path.home() / 'lite3_ws/log/operator/state.env')
    p.add_argument('--sample-interval', type=float, default=5.0, help='Seconds between saved frames')
    args=p.parse_args()
    args.rgbd = args.camera_mode != 'wide'
    wide_search = args.camera_mode == 'wide_search'
    if not 0 < args.scout_fps < float("inf"):
        p.error('--scout-fps must be finite and positive')
    if args.locate_wide_qr and not wide_search:
        p.error('--locate-wide-qr requires --wide-search')
    profile = nav_run = None
    if args.competition is not None:
        if args.resume or args.objectives or not args.rgbd:
            p.error('--competition requires RGB-D and a fresh database with tasks from its profile')
        try:
            profile = validate_profile(read_json(args.competition))
            if profile["mode"] == "competition":
                nav_run = prepare_navigation(args.navigation_state.expanduser())
        except (KeyError, OSError, ValueError) as exc:
            p.error(f'Competition setup: {exc}')
    if args.scene_understanding:
        if not args.rgbd or profile is None or profile.get("contract") != "sutd-v6.3":
            p.error('--scene-understanding requires --rgbd or --wide-search and a v6.3 scoring profile')
        if args.dry_run:
            p.error('--scene-understanding sends camera frames to Claude; remove --dry-run to enable it')
        from scene_scoring import SceneScoring
    if args.publish_mission_events and not args.rgbd:
        p.error('--publish-mission-events requires RGB-D; remove --wide')
    if not 0 < args.sample_interval < float('inf'):
        p.error('--sample-interval must be finite and positive')
    if not 0<args.confidence<=1 or args.threads<1 or args.imgsz<32 or args.seconds<0:
        p.error('Invalid confidence, thread count, image size or duration')
    if not Path(args.model).expanduser().is_file():
        p.error('Model file not found; supply an existing .pt file')
    if args.max_requests<0 or args.cooldown<0 or args.min_area<1:
        p.error('Invalid trigger limits')
    if args.scene_understanding or (args.vlm and not args.dry_run):
        key=os.environ.get('ANTHROPIC_API_KEY','').strip()
        if not key or not key.isascii() or any(c.isspace() for c in key):
            p.error('Set ANTHROPIC_API_KEY locally or use the launcher hidden prompt')
        os.environ['ANTHROPIC_API_KEY']=key
    if args.resume is not None and not args.resume.expanduser().is_file():
        p.error(f'Cannot resume a missing database: {args.resume}')

    started_at = datetime.now(timezone.utc)
    libraries = ROOT / 'data' / 'runs'
    libraries.mkdir(parents=True, exist_ok=True)
    prefix = started_at.strftime('%Y%m%dT%H%M%S_%fZ-')
    folder = Path(mkdtemp(prefix=prefix, dir=libraries))
    db_path = (args.resume or args.db or folder / 'mission.sqlite3').expanduser().resolve()
    if args.db is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            db_path.touch(exist_ok=False)
        except FileExistsError:
            p.error(f'Database already exists: {db_path}. Choose a new path or use --resume.')

    with LiveStore(db_path, args.sample_interval, args.objectives):
        pass
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    model=YOLO(str(Path(args.model).expanduser()))
    print('Classes:',json.dumps(model.names),flush=True)
    camera_description = 'Wide search with Orbbec location' if wide_search else (
        'ROS RGB-D' if args.rgbd else 'Front RTSP camera'
    )
    print(camera_description + ' -> YOLO + QR + yellow triangles. Q quit | S save evidence.', flush=True)
    print('Pretrained weights do not include curtain/staircase. Use --model for custom weights.',flush=True)
    person_reid = None
    if args.reid:
        from person_reid import PersonReID
        person_reid = PersonReID(ROOT / 'models' / 'osnet_x0_25_msmt17.pt', args.confidence)
    if args.rgbd:
        from rgbd_camera import RGBDCamera
        camera = RGBDCamera(publish_mission_events=args.publish_mission_events,
                             receive_validations=profile is not None, on_demand=wide_search)
    else:
        camera = Camera()
    if profile is not None:
        try:
            origin = camera.start_pose()
            deadline = time.monotonic() + (8 if nav_run or args.scene_understanding else 1)
            while origin is None and time.monotonic() < deadline:
                time.sleep(0.05)
                origin = camera.start_pose()
            if nav_run:
                if origin is None:
                    raise ValueError('No fresh map-to-base_link pose at the start cell')
                with (nav_run / 'live_mission_competition.attempt').open('x', encoding='utf-8') as claim:
                    claim.write(str(db_path))
            with LiveStore(db_path, args.sample_interval) as ready:
                Competition.initialize(ready.store, profile, origin, nav_run)
        except (OSError, ValueError, KeyboardInterrupt) as exc:
            camera.close()
            p.error(f'Competition setup: {exc}')
    database_pointer = folder / 'database.path.tmp'
    database_pointer.write_text(str(db_path), encoding='utf-8')
    database_pointer.replace(folder / 'database.path')
    print('Run library:', folder, flush=True)
    print('Mission database:', db_path, flush=True)
    room_camera = Camera() if wide_search else None
    perception_camera = room_camera if wide_search else camera
    if room_camera is not None:
        print('Wide camera room view with an RGB-D inset.', flush=True)
    stop=threading.Event();lock=threading.Lock()

    def detect_objects(frame):
        result = model.predict(frame, imgsz=args.imgsz, conf=args.confidence, device='cpu', verbose=False)[0]
        return [dict(label=result.names[int(box.cls.item())], confidence=float(box.conf.item()),
                     xyxy=box.xyxy[0].tolist()) for box in result.boxes]

    handoff = WideHandoff(
        camera, detect_objects, detect_qr, stop, locate_qr=args.locate_wide_qr,
        detect_signs=lambda frame: detect_yellow_triangles(frame, args.min_area),
    ) if wide_search else None
    state={'result':None,'count':0,'error':None}
    tracker=SignTracker()
    pool=ThreadPoolExecutor(max_workers=1);future=None;seen=[];last_sent=-1e9
    sent=0;handled=None;labels={};vlm_status='VLM off'
    print('Room classification:', 'enabled' if args.scene_understanding else 'off', flush=True)
    print('VLM mode:', 'local trigger dry-run' if args.dry_run else 'enabled' if args.vlm else 'off',flush=True)
    def infer():
        last = None
        next_scout = 0.0
        recorder = scene_scoring = None
        try:
            recorder = LiveStore(
                db_path, args.sample_interval, person_reid=person_reid,
                on_observation=camera.publish_candidate if args.rgbd else None,
                on_event=camera.publish_event if args.publish_mission_events else None,
            )
            competition = Competition(recorder.store) if profile is not None else None
            if args.scene_understanding:
                scene_scoring = SceneScoring(competition, args.vlm_model, args.max_requests, args.cooldown)
            clue_count = len(recorder.qr_records)
            if competition is not None:
                competition.collect_qr()
            while not stop.is_set():
                if scene_scoring is not None:
                    scene_scoring.poll()
                if competition is not None:
                    for _ in range(8):
                        validation = camera.next_validation()
                        if validation is None:
                            break
                        try:
                            competition.ingest(validation)
                        except (KeyError, TypeError, ValueError) as exc:
                            print(f'Rejected scoring validation: {exc}', flush=True)
                sample=perception_camera.latest()
                if sample is None or sample[0]==last:
                    stop.wait(.005);continue
                received,utc,frame=sample[:3];last=received
                if wide_search and received < next_scout:
                    stop.wait(.01)
                    continue
                if time.monotonic()-received>1:
                    stop.wait(.02);continue
                started=time.monotonic()
                if wide_search:
                    next_scout = received + 1.0 / args.scout_fps
                rgbd_detection = args.rgbd and not wide_search
                depth_pair = camera.pair_depth(sample[3]) if rgbd_detection else None
                qr_codes = detect_qr(frame)
                boxes = detect_objects(frame)
                if rgbd_detection:
                    pending_qr = [qr for qr in qr_codes if recorder.qr_needs_location(qr)]
                    camera.locate(boxes, sample[3], depth_pair, qr_codes=pending_qr)
                yellow_signs=tracker.update(detect_yellow_triangles(frame,args.min_area),time.monotonic())
                if wide_search:
                    boxes.extend(dict(label="hazard_sign", confidence=0.0, xyxy=sign["box"])
                                 for sign in yellow_signs if sign["hits"] >= 3)
                record=dict(received_at_utc=utc,timestamp_source='ROS color image header' if rgbd_detection else 'local RTSP arrival',
                    frame_received=received,processing_ms=round((time.monotonic()-started)*1000,1),
                    image_width=frame.shape[1],image_height=frame.shape[0],objects=boxes,qr_codes=qr_codes,yellow_sign_candidates=yellow_signs)
                if wide_search:
                    record['camera_role'] = 'wide_search_detection'
                observations = recorder.record(frame, record, handoff=handoff)
                if scene_scoring is not None:
                    scene_scoring.offer(frame, observations)
                if competition is not None and len(recorder.qr_records) != clue_count:
                    clue_count = len(recorder.qr_records)
                    competition.collect_qr()
                with lock:
                    state['result']=(frame,record);state['count']+=1
        except Exception as exc:
            with lock:state['error']=f'{type(exc).__name__}: {exc}'
        finally:
            if scene_scoring is not None:
                scene_scoring.close()
            if recorder is not None:
                recorder.close()
    worker=threading.Thread(target=infer,daemon=True);worker.start()
    started=time.monotonic();last_print=started;last_count=0;fps=0.
    try:
        while not args.seconds or time.monotonic()-started<args.seconds:
            if perception_camera.error:
                raise RuntimeError(f'Camera: {perception_camera.error}')
            with lock:
                output=state['result'];count=state['count'];error=state['error']
            if error:raise RuntimeError(error)
            now=time.monotonic()
            if future is not None and future.done():
                response=future.result();future=None
                print('VLM result:',json.dumps(response),flush=True)
                for entry in response.get('result',{}).get('signs',[]):labels[entry['candidate_id']]=entry
                vlm_status='Identification received' if response['success'] and not args.dry_run else 'Trigger saved locally' if args.dry_run else 'VLM failed; see terminal'
                if response.get('status_code') in (400,401,403,404):args.vlm=False
            if output and handled!=output[1]['frame_received']:
                handled=output[1]['frame_received']
                seen=[(when,f) for when,f in seen if now-when<args.cooldown]
                eligible=[]
                for sign in output[1]['yellow_sign_candidates']:
                    if sign['hits']<3:continue
                    crop=crop_sign(output[0],sign)
                    if min(crop.shape[:2])<40:continue
                    f=fingerprint(crop)
                    if any(np.count_nonzero(f!=other)<=8 for _,other in seen):continue
                    eligible.append((sign,f))
                permitted=not args.max_requests or sent<args.max_requests
                fresh=now-output[1]['frame_received']<1
                if eligible and future is None and permitted and fresh and now-last_sent>=5 and (args.vlm or args.dry_run):
                    chosen=eligible[:4]
                    future=pool.submit(identify,output[0].copy(),output[1]['received_at_utc'],[v[0] for v in chosen],args.vlm_model,folder,args.dry_run)
                    seen.extend((now,f) for _,f in chosen);last_sent=now;sent+=1
                    vlm_status='Identifying signs...' if not args.dry_run else 'Saving trigger...'
                    print('VLM trigger:',[v[0]['id'] for v in chosen],flush=True)
            if args.max_requests and sent>=args.max_requests and future is None:
                vlm_status=f'Request limit reached: {sent}; local detection continues'
            if now-last_print>=2:
                fps=(count-last_count)/(now-last_print)
                if output:
                    record=output[1]
                    print(json.dumps(dict(inference_fps=round(fps,1),processing_ms=record['processing_ms'],
                        objects=record['objects'],qr_codes=record['qr_codes'])),flush=True)
                last_print=now;last_count=count
            if args.headless:
                time.sleep(.01)
                continue
            sample = perception_camera.latest()
            frame = (output[0] if output else sample[2]).copy() if sample is not None else None
            if frame is None and room_camera is None:
                time.sleep(.01)
                continue
            if frame is not None:
                # Draw detections on their own camera frame.
                if output:
                    for b in output[1]['objects']:
                        x1,y1,x2,y2=map(int,b['xyxy'])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,f"{b['label']} {b['confidence']:.2f}",
                            (x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
                    for sign in output[1].get('yellow_sign_candidates',[]):
                        sx1,sy1,sx2,sy2=sign['box']
                        cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),(0,255,255),2)
                        label=f"#{sign['id']} possible warning sign"
                        entry=labels.get(sign['id'])
                        if entry:
                            label=f"#{sign['id']} "+(entry['hazard_label'] if entry['is_warning_sign'] else 'Not a warning sign')+(' ?' if entry['uncertain'] else '')
                        cv2.putText(frame,label,(sx1,max(75,sy1-8)),0,.6,(0,255,255),2)
                    for qr in output[1]['qr_codes']:
                        polygon=np.asarray(qr['corners'],dtype='int32')
                        cv2.polylines(frame,[polygon],True,(255,255,0),3)
                        label=('QR: '+qr['text'][:70]) if qr['decoded'] else 'QR located - not decoded'
                        cv2.putText(frame,label,(max(0,int(polygon[0,0])),max(50,int(polygon[0,1])-10)),0,.65,(255,255,0),2)
                cv2.putText(frame,f'YOLO {fps:.1f} inference FPS | Q quit | S save',
                    (15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
                cv2.putText(frame,f'{vlm_status} | sign requests {sent}',(15,58),0,.6,(0,255,255),2)
            if room_camera is not None:
                wide_sample = room_camera.latest()
                if output and frame is not None and now - output[1]['frame_received'] < 0.5:
                    wide_sample = (output[1]['frame_received'], output[1]['received_at_utc'], frame)
                rgbd_sample = camera.latest()
                rgbd_view = rgbd_sample[2] if rgbd_sample is not None else None
                cv2.imshow('Lite3 live mission', room_preview(
                    wide_sample, rgbd_view if not camera.error else None, now))
            else:
                cv2.imshow('Lite3 live mission', cv2.resize(frame, (960, 540)))
            key=cv2.waitKey(1)&255
            if key==ord('q'):break
            if key==ord('s') and output and frame is not None:
                path=folder/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                path.mkdir()
                # Save the exact inference frame, not a newer preview frame.
                cv2.imwrite(str(path/'frame.png'),output[0])
                record=dict(output[1]);record.pop('frame_received',None);record['sign_identifications']=dict(labels)
                cv2.imwrite(str(path/'annotated.jpg'),frame)
                (path/'detections.json').write_text(json.dumps(record,indent=2))
                print('Saved:',path,flush=True)
            time.sleep(.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if room_camera is not None:
            room_camera.stop.set()
        worker.join(timeout=55 if args.scene_understanding else 15)
        if room_camera is not None:
            room_camera.thread.join(timeout=6)
        if args.rgbd:
            camera.close()
        else:
            camera.stop.set();camera.thread.join(timeout=6)
        if future is not None:print('Finishing current API request; no new requests.',flush=True)
        pool.shutdown(wait=True)
        if not args.headless:cv2.destroyAllWindows()
        with lock:output=state['result'];count=state['count'];error=state['error']
        if args.headless and output:
            cv2.imwrite(str(folder/'latest_test.png'),output[0])
            record=dict(output[1]);record.pop('frame_received',None);record['sign_identifications']=dict(labels)
            (folder/'latest_test.json').write_text(json.dumps(record,indent=2))
        print('Run saved:', folder, flush=True)
        if error:
            raise RuntimeError(error)
        print(f'Finished: {count} inference frames in {time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':main()
