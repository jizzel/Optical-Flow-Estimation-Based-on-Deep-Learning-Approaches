from pathlib import Path

import numpy as np

path = Path('out.flo')
with path.open(mode='r') as flo:
    np_flow = np.fromfile(flo, np.float32)
    print(np_flow.shape)

with path.open(mode='r') as flo:
  tag = np.fromfile(flo, np.float32, count=1)[0]
  width = np.fromfile(flo, np.int32, count=1)[0]
  height = np.fromfile(flo, np.int32, count=1)[0]
  print('test ', np.fromfile(flo, np.float32, count=1)[0])
  print('test2 ', np.fromfile(flo, np.float32, count=1)[0])
  print('test1 ', np.fromfile(flo, np.float32, count=1)[0])
  print('test12 ', np.fromfile(flo, np.float32, count=1)[0])

  print('tag', tag, 'width', width, 'height', height, '\n')

  nbands = 2
  tmp = np.fromfile(flo, np.float32, count= nbands * width * height)
  flow = np.resize(tmp, (int(height), int(width), int(nbands)))

print('flow: ', flow[0][1])
print('flow shape: ', flow.shape)

# a='/images/frame_000'
# b='.png'
# output='out.flo'
# for i in range(50):
#   python run.py --model sintel-final --first a+i+b --second ./images/second.png --out ./output
#   i=i+2
# done
