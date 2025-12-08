import json
import html_maker
from pathlib import Path
import argparse
from os import makedirs
from shutil import copyfile
from tqdm import tqdm
from math import isnan
from concurrent.futures import ProcessPoolExecutor


IMAGE_SUFFIXES=['.jpg', '.png']

def find_existing_image_path(dir_path, image_name):
    dir_path = Path(dir_path)
    for suffix in IMAGE_SUFFIXES:
        if (dir_path / f'{image_name}{suffix}').exists():
            return str(relative_path(dir_path / f'{image_name}{suffix}'))
    raise BaseException(f'{image_name} doesnt exists in {str(dir_path)}')

def relative_path(path, parent_num=1):
    return str(path.relative_to(path.parents[parent_num]))

def arrange_results(json_data, dir_dict, title, save_dir, category=None):
    if json_data is None and 'results' in dir_dict:
        json_data = {str(im_path.name): None for im_path in dir_dict['results'].iterdir()}
    elif json_data is None and 'renders' in dir_dict:
        json_data = {str(im_path.name): None for im_path in dir_dict['renders'].iterdir()}
    elif json_data is None:
        raise ValueError('no json data and no results or render folders')
    num_results = 0
    sum_result = 0.0
    column_list = []
    for ind, (image_name, image_result) in tqdm(enumerate(json_data.items())):
        image_name = Path(image_name).name
        image_stem = str(Path(image_name).stem)

        if image_result is not None and image_result != 'Nan' and not isnan(image_result):
            sum_result += float(image_result)
            num_results += 1

        if ind < 30:
            section_list = []
            label_text = f'{image_stem}: {image_result * 100 :.2f}%' if image_result is not None \
                else f'{image_stem}:'
            section_list.append(html_maker.LabelView(label_text))
            # results, gt_results, pca, gt_pca, renders, gt_rgb
            if 'results' in dir_dict:
                first_row = get_first_row(dir_dict, image_name, save_dir / category)
                section_list.append(first_row)
            
            second_row = get_second_row(dir_dict, image_stem, save_dir)
            section_list.append(second_row)

            third_row = get_third_row(dir_dict, image_name, save_dir / category)
            if third_row:
                section_list.append(third_row)
            
            column_list.append(html_maker.SectionView(section_list))
    
    title_text = f'{title}: AP: {100 * sum_result / num_results :.2f}%' if num_results \
        else title
    column_list = [html_maker.LabelView(title_text, size='24px')] \
                   + column_list

    data=html_maker.ColumnView(column_list)
    return data

def get_second_row(dir_dict, image_stem, save_dir):
    second_row = []
    if 'gt_rgb' in dir_dict:
        if not (save_dir / 'gt_rgb' / f'{image_stem}.png').exists():
            copyfile(str(dir_dict['gt_rgb'] / f'{image_stem}.png'),
                    str(save_dir / 'gt_rgb' / f'{image_stem}.png'))
        second_row.append(html_maker.ColumnView([
                    html_maker.LabelView('GT RGB'),
                    html_maker.ImageView(str(Path('gt_rgb')
                                            / f'{image_stem}.png'))
                ]))
    if 'renders' in dir_dict:
        if not (save_dir / 'renders' / f'{image_stem}.png').exists():
            copyfile(str(dir_dict['renders'] / f'{image_stem}.png'),
                    str(save_dir / 'renders' / f'{image_stem}.png'))
        second_row.append(html_maker.ColumnView([
                    html_maker.LabelView('Rendered RGB'),
                    html_maker.ImageView(str(Path('renders')
                                            / f'{image_stem}.png'))
                ]))
    if 'gt_pca' in dir_dict:
        if not (save_dir / 'gt_pca' / f'{image_stem}_feature_vis.png').exists():
            copyfile(str(dir_dict['gt_pca'] / f'{image_stem}_feature_vis.png'),
                    str(save_dir / 'gt_pca' / f'{image_stem}_feature_vis.png'))
        second_row.append(html_maker.ColumnView([
                    html_maker.LabelView('2D PCA'),
                    html_maker.ImageView(str(Path('gt_pca')
                                            / f'{image_stem}_feature_vis.png'))
                ]))
    if 'pca' in dir_dict:
        if not (save_dir / 'pca' / f'{image_stem}_feature_vis.png').exists():
            copyfile(str(dir_dict['pca'] / f'{image_stem}_feature_vis.png'),
                    str(save_dir / 'pca' / f'{image_stem}_feature_vis.png'))
        second_row.append(html_maker.ColumnView([
                    html_maker.LabelView('PCA'),
                    html_maker.ImageView(str(Path('pca')
                                            / f'{image_stem}_feature_vis.png'))
                ]))
    
    return html_maker.RowView(second_row)

def get_first_row(dir_dict, image_name, save_dir):
    first_row = []
    filename = image_name.replace('.jpg', '_vis.jpg')
    if 'gt_results' in dir_dict:
        copyfile(str(dir_dict['gt_results'] / filename),
                 str(save_dir / 'gt_results' / filename))
        first_row.append(html_maker.ColumnView([
                html_maker.LabelView('GT Result'),
                html_maker.ImageView(str(Path(save_dir.name) / 'gt_results' / filename))
            ]))
        
    copyfile(str(dir_dict['results'] / filename),
                 str(save_dir / 'results' / filename))
    first_row.append(html_maker.ColumnView([
            html_maker.LabelView('Result'),
            html_maker.ImageView(str(Path(save_dir.name) / 'results' / filename))
        ]))
    
    prec_recall = image_name.replace('.jpg', '_prec_recall.png')
    if (dir_dict['results'] / prec_recall).exists():
        copyfile(str(dir_dict['results'] / prec_recall),
                 str(save_dir / 'results' / prec_recall))
        first_row.append(html_maker.ColumnView([
                html_maker.LabelView('Precision/Recall'),
                html_maker.ImageView(str(Path(save_dir.name) / 'results' / prec_recall))
            ]))
    return html_maker.RowView(first_row)

def find_file(base_path: Path) -> Path:
    """Find an existing file by checking all possible suffixes."""
    for file in base_path.parent.glob(base_path.name + "*_*"):
        if file.is_file():
            return file  # Return the first found file
    return None

def get_third_row(dir_dict, image_name, save_dir):
    third_row = []
    
    for val in [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]:
        filename = image_name.replace('.jpg', f"_recall_{str(val).replace('.', '')}")
        path = find_file(dir_dict['results'] / filename)

        if path is not None:
            precision = float(path.stem.split('_')[-1]) / 1000 * 100
            recall = val * 100
            copyfile(str(path),
                    str(save_dir / 'results' / path.name))
            third_row.append(html_maker.ColumnView([
                    html_maker.LabelView(f'Recall {recall}%: Precision {precision: .2f}%'),
                    html_maker.ImageView(str(Path(save_dir.name) / 'results' / path.name))
                ]))
    return html_maker.RowView(third_row)

def build_first_line(dir_dict, image_name, image_results):
    match_name = image_name.replace('real', 'render')
    loss = None
    for res in image_results['base_images']:
        if res['image'] == match_name:
            loss = res['loss']
    
    row_list = [html_maker.SectionView([
                    html_maker.LabelView(image),
                    html_maker.ImageView(
                        find_existing_image_path(dir_dict['images'], image)),
                    html_maker.ImageView(
                        find_existing_image_path(dir_dict['pca'], image)),
                    html_maker.LabelView(bot_text)])
                for image, bot_text in ((image_name, ''), (match_name, f'Loss: {loss: .5}'))
                ]
    if 'loss2D' in dir_dict:
        row_list.append(
            html_maker.SectionView([html_maker.LabelView('Loss2D'),
                                    html_maker.ImageView(
                                         find_existing_image_path(dir_dict['loss2D'], f'{image_name}_{match_name}'))]))
    return html_maker.RowView(row_list)

def add_if_exists(dir_dict, key, path):
    if path.exists():
        dir_dict[key] = path

def handle_category(args, renders_path, pca_path, gt_rgb_path, gt_pca_path,
                    save_dir, data, categories_paths, ind, res_path):
    print(f'Preparing Visualization for {res_path.name} ({ind} / {len(categories_paths)}) ...')
    dir_dict = {
                'results': res_path            
            }
    if args.gt_dir:
        if (Path(args.gt_dir) / res_path.name).exists():
            dir_dict['gt_results'] = Path(args.gt_dir) / res_path.name
        else:
            dir_dict['gt_results'] = Path(args.gt_dir) / res_path.name.split(',')[0]
        if not dir_dict['gt_results'].exists():
            raise ValueError(f"{dir_dict['gt_results']} doesn't exists")
    
    add_if_exists(dir_dict, 'pca', pca_path)
    add_if_exists(dir_dict, 'renders', renders_path)
    add_if_exists(dir_dict, 'gt_rgb', gt_rgb_path)
    add_if_exists(dir_dict, 'gt_pca', gt_pca_path)

    for key in dir_dict.keys():
        if 'results' in key:
            makedirs(str(save_dir / res_path.name / key), exist_ok=True)
        else:
            makedirs(str(save_dir / key), exist_ok=True)
            
    html_output_path = save_dir / f'results_{res_path.name}.html'
            
    data_class = data[res_path.name] if data is not None else None
    data_to_present = arrange_results(data_class, dir_dict, res_path.name, save_dir, res_path.name)
            
    html = html_maker.generate_html(data_to_present)

    with open(str(html_output_path), 'w') as f:
        f.write(html)

def handle_category_wrapper(multi_args):
    handle_category(*multi_args)


def arrange_pred_only_results(json_data, dir_dict, title, save_dir, category=None):
    json_data = {str(im_path.name.replace('_vis', '')): None for im_path in dir_dict['results'].iterdir()
                 if im_path.stem.endswith('_vis')}
    
    num_results = 0
    sum_result = 0.0
    column_list = []
    for ind, (image_name, image_result) in tqdm(enumerate(json_data.items())):
        image_name = Path(image_name).name
        image_stem = str(Path(image_name).stem)

        if image_result is not None and image_result != 'Nan' and not isnan(image_result):
            sum_result += float(image_result)
            num_results += 1

        if ind < 30:
            section_list = []
            label_text = f'{image_stem}: {image_result * 100 :.2f}%' if image_result is not None \
                else f'{image_stem}:'
            section_list.append(html_maker.LabelView(label_text))
            # results, gt_results, pca, gt_pca, renders, gt_rgb
            if 'results' in dir_dict:
                first_row = get_first_row(dir_dict, image_name, save_dir / category)
                section_list.append(first_row)
            
            second_row = get_second_row(dir_dict, image_stem, save_dir)
            section_list.append(second_row)

            third_row = get_third_row(dir_dict, image_name, save_dir / category)
            if third_row:
                section_list.append(third_row)
            
            column_list.append(html_maker.SectionView(section_list))
    
    title_text = f'{title}: AP: {100 * sum_result / num_results :.2f}%' if num_results \
        else title
    column_list = [html_maker.LabelView(title_text, size='24px')] \
                   + column_list

    data=html_maker.ColumnView(column_list)
    return data

def handle_pred_only_category_wrapper(multi_args):
    handle_pred_only_category(*multi_args)

def handle_pred_only_category(args, renders_path, pca_path, gt_rgb_path, gt_pca_path,
                              save_dir, data, categories_paths, ind, res_path):
    print(f'Preparing Visualization for {res_path.name} ({ind} / {len(categories_paths)}) ...')
    dir_dict = {
                'results': res_path            
            }
    
    add_if_exists(dir_dict, 'renders', renders_path)
    add_if_exists(dir_dict, 'gt_rgb', gt_rgb_path)

    for key in dir_dict.keys():
        if 'results' in key:
            makedirs(str(save_dir / res_path.name / key), exist_ok=True)
        else:
            makedirs(str(save_dir / key), exist_ok=True)
            
    html_output_path = save_dir / f'results_{res_path.name}.html'
            
    data_class = data[res_path.name] if data is not None else None
    data_to_present = arrange_pred_only_results(data_class, dir_dict, res_path.name, save_dir, res_path.name)
            
    html = html_maker.generate_html(data_to_present)

    with open(str(html_output_path), 'w') as f:
        f.write(html)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("run_dir", type=str, help="Directory of the run")
    parser.add_argument("--run_name", type=str, default='ours_70000',
                        help="Run name")
    parser.add_argument("--gt_dir", type=str, default='',
                        help="GT dir")
    parser.add_argument("--vis_dir", type=str, default='vis_saved_3d_feature',
                        help="GT dir")
    parser.add_argument("--only_preds", type=bool, default=False,
                        help="No GT comparison")
    args = parser.parse_args()

    base_path = Path(args.run_dir)
    result_path = base_path / args.vis_dir
    renders_path = base_path / 'train' / args.run_name / 'renders'
    pca_path = base_path / 'train' / args.run_name / 'feature_map'
    gt_rgb_path = base_path / 'train' / args.run_name / 'gt'
    gt_pca_path = base_path / 'train' / args.run_name / 'gt_feature_map'
    save_dir = base_path / 'htmls' / f'{args.run_name}_{args.vis_dir}'
    makedirs(save_dir, exist_ok=True)

    json_file = result_path / 'data.json'
    if not args.only_preds and Path(json_file).exists():
        with open(str(json_file), 'r') as file:
            data = json.load(file)
    else:
        data = None

    if not args.only_preds and result_path.exists():
        multi_args = []
        if json_file.exists():
            with open(str(json_file)) as file:
                categories = json.load(file).keys()
            categories_paths = [Path(result_path / cat) for cat in categories]
        else:
            categories_paths = [path for path in result_path.iterdir() if path.is_dir()]
            
        for ind, res_path in enumerate(categories_paths):
            multi_args.append((args, renders_path, pca_path, gt_rgb_path, gt_pca_path,
                               save_dir, data, categories_paths, ind, res_path))
            # handle_category(args, renders_path, pca_path, gt_rgb_path, gt_pca_path,
            #                 save_dir, data, categories_paths, ind, res_path)

        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(handle_category_wrapper, multi_args), total=len(multi_args)))
    elif result_path.exists():
        multi_args = []
        categories_paths = [path for path in result_path.iterdir() if path.is_dir()]
        for ind, res_path in enumerate(categories_paths):
            multi_args.append((args, renders_path, pca_path, gt_rgb_path, gt_pca_path,
                               save_dir, data, categories_paths, ind, res_path))
            # handle_pred_only_category(args, renders_path, pca_path, gt_rgb_path, gt_pca_path,
            #                           save_dir, data, categories_paths, ind, res_path)
            
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(handle_pred_only_category_wrapper, multi_args), total=len(multi_args)))
    else:
        dir_dict = {}
        add_if_exists(dir_dict, 'pca', pca_path)
        add_if_exists(dir_dict, 'renders', renders_path)
        add_if_exists(dir_dict, 'gt_rgb', gt_rgb_path)
        add_if_exists(dir_dict, 'gt_pca', gt_pca_path)
        
        for key in dir_dict.keys():
            makedirs(str(save_dir / key), exist_ok=True)

        html_output_path = save_dir / f'results.html'
        
        data_class = None
        data_to_present = arrange_results(data_class, dir_dict, 'Train Results', save_dir)
        
        html = html_maker.generate_html(data_to_present)

        with open(str(html_output_path), 'w') as f:
            f.write(html)
