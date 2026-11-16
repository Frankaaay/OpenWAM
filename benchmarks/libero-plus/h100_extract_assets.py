import concurrent.futures,hashlib,json,os,shutil,stat,time,zipfile
from pathlib import Path,PurePosixPath
root=Path('/gpfs/jiuquyun/projects/Frank/openwam-libero-plus-20261116/frank/runs/libero-plus-pro-20261116/assets-plus')
archive=root/'assets.zip'
h=hashlib.sha256()
with archive.open('rb') as f:
 for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
assert h.hexdigest()=='96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf'
print('PLUS_ARCHIVE_VERIFIED',h.hexdigest(),flush=True)
target=root/'extracted/assets'
target.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
resume_path=root/'extraction-resume.json'
resume=json.loads(resume_path.read_text()) if resume_path.exists() else None
if resume:assert resume['archive_sha256']==h.hexdigest()
with zipfile.ZipFile(archive) as z:
 files=[];dirs={PurePosixPath('.')};names=set()
 for info in z.infolist():
  parts=PurePosixPath(info.filename).parts
  assert 'assets' in parts and '..' not in parts and not PurePosixPath(info.filename).is_absolute()
  assert not stat.S_ISLNK(info.external_attr>>16)
  rel=PurePosixPath(*parts[parts.index('assets')+1:])
  if info.is_dir():dirs.add(rel);continue
  assert rel not in names;names.add(rel)
  files.append((info,rel))
  dirs.update(rel.parents)
 # The public archive embeds its author's machine prefix. Strip only that
 # prefix, retaining every asset and checking CRC while streaming each file.
 if not resume:
  for d in sorted(dirs,key=lambda p:(len(p.parts),str(p))):(target/d).mkdir(exist_ok=True)
 def write(item):
  rel,data=item
  with (target/rel).open('wb') as dst:dst.write(data)
  assert (target/rel).stat().st_size==len(data)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  batch=[];batch_bytes=0;done=resume['completed_files'] if resume else 0
  print('RESUME_EXTRACTION',done,flush=True)
  for info,rel in files[done:]:
   # Serial archive reads: concurrent seek/read on this FUSE mount caused
   # decompression errors; the same first 2000 entries passed serial CRC.
   try:data=z.read(info)
   except Exception as exc:raise RuntimeError(f'Invalid archive member: {info.filename}') from exc
   batch.append((rel,data));batch_bytes+=len(data)
   if len(batch)>=2000 or batch_bytes>=64*1024*1024:
    list(pool.map(write,batch));done+=len(batch)
    print('EXTRACTED',done,len(files),round(time.monotonic()-start,1),flush=True)
    batch=[];batch_bytes=0
  list(pool.map(write,batch));done+=len(batch)
report=dict(complete=True,archive_sha256=h.hexdigest(),files=len(files),directories=len(dirs),bytes=sum(x.file_size for x,_ in files),seconds=time.monotonic()-start)
(root/'ready.json').write_text(json.dumps(report,indent=2))
print('PLUS_ASSETS_READY',json.dumps(report),flush=True)
