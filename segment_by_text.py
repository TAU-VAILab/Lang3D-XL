from pathlib import Path
import argparse
import numpy as np
from os import makedirs
from PIL import Image
from lang_sam import LangSAM

def main(images_dir, save_dir, text_prompt, debug=False):
    model = LangSAM()
    makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)
    images_dir = Path(images_dir)
    if debug:
        image_path = Path(r'/storage/shai/3d/data/rgb_data/hurba/building_segmentation/0065.jpg')
        segment_image_by_text(text_prompt, model, image_path, save_dir / image_path.name)
    for image_path in [path for path in images_dir.iterdir() if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')]:
        segment_image_by_text(text_prompt, model, image_path, save_dir / image_path.name)

def segment_image_by_text(text_prompt, model, image_path, save_path):
    image_pil = Image.open(image_path).convert("RGB")
    results = []
    w, h = image_pil.size
    mask = np.zeros((h, w))
    for prompt in text_prompt.split(','):
        results = model.predict([image_pil], [prompt])
        if len(results) > 0 and len(results[0]['masks']) > 0:
            mask = results[0]['masks'][0] * results[0]['scores'][0]
            for ind, m in enumerate(results[0]['masks'][1:]):
                mask[m > 0] = np.maximum(mask[m > 0], results[0]['scores'][ind])
            # image = np.array(image_pil)
    Image.fromarray(np.uint8(mask * 255)).save(save_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("images_dir", type=str, help="Directory of the images")
    parser.add_argument("save_dir", type=str, help="Directory to save the segmentation")
    parser.add_argument("text_prompt", type=str, help="Text to segment by")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    main(args.images_dir, args.save_dir, args.text_prompt, args.debug)
    # main('/storage/shai/3d/data/rgb_data/blue_mosque/images', '/storage/shai/3d/data/rgb_data/blue_mosque/test_langsam',
    #      'domes', False)
