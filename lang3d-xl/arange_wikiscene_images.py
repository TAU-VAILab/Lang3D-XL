from pathlib import Path
from os import makedirs
from shutil import copyfile

main_path = Path(r'/root/feature-3dgs/data/berlin_cathedral')
dir_num = '5'
sub_dir = '1'
image_suffixes = ('.jpg', '.JPG', '.png', '.PNG')

makedirs(str(main_path / 'input'), exist_ok=True)
makedirs(str(main_path / 'sparse'), exist_ok=True)
makedirs(str(main_path / 'sparse' / '0'), exist_ok=True)

with open(str(main_path / f'{dir_num}_3d' / sub_dir / 'images.txt')) as file:
    lines = file.read().split('\n')
    new_lines = []
    image_num = 1
    skip = False
    for i, line in enumerate(lines):
        words = line.split(' ')
        should_print = False
        if words[-1].endswith(image_suffixes):
            # print(i, words[-1].split('/')[-1])
            suffix = words[-1].split('.')[-1]
            new_name = f'image_{image_num}.{suffix}'
            if (main_path / words[-1]).exists():
                copyfile(str(main_path / words[-1]), str(main_path / 'input' / new_name))
                print(f'coppied {words[-1]} to {new_name}')
                words[-1] = new_name
                image_num += 1
                new_lines.append(' '.join(words))
            else:
                skip = True
        elif skip:
            skip = False
        else:
            new_lines.append(' '.join(words))

with open(str(main_path / 'sparse' / '0' / 'images.txt'), 'w') as file:
    file.write('\n'.join(new_lines))

copyfile(str(main_path / f'{dir_num}_3d' / sub_dir / 'cameras.txt'), str(main_path / 'sparse' / '0' / 'cameras.txt'))
copyfile(str(main_path / f'{dir_num}_3d' / sub_dir / 'points3D.txt'), str(main_path / 'sparse' / '0' / 'points3D.txt'))
