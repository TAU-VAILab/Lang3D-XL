import json
import html_maker
from pathlib import Path 


IMAGE_SUFFIXES=['.jpg', '.png']

def find_existing_image_path(dir_path, image_name):
    dir_path = Path(dir_path)
    for suffix in IMAGE_SUFFIXES:
        if (dir_path / f'{image_name}{suffix}').exists():
            return str(relative_path(dir_path / f'{image_name}{suffix}'))
    raise BaseException(f'{image_name} doesnt exists in {str(dir_path)}')

def relative_path(path, parent_num=1):
    return str(path.relative_to(path.parents[parent_num]))

def arrange_results(json_file, dir_dict, title):
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    num_all_correct = 0
    num_render_correct = 0
    num_base_correct = 0
    column_list = []
    for image_name, image_results in json_data.items():
        is_all_correct = image_name.replace('real', 'render') == Path(image_results['all_images'][0]['image']).stem
        is_base_correct = image_name.replace('real', 'render') == Path(image_results['base_images'][0]['image']).stem
        
        render_only_results = [image_res for image_res in image_results['all_images'] if 'render' in image_res['image']]
        is_render_correct = image_name.replace('real', 'render') == Path(render_only_results[0]['image']).stem
        
        num_all_correct += int(is_all_correct)
        num_base_correct += int(is_base_correct)
        num_render_correct += int(is_render_correct)

        section_list = []
        x = build_first_line(dir_dict, image_name, image_results)
        section_list.append(x)

        main_pca_view = html_maker.ImageView(
            find_existing_image_path(dir_dict['pca'], image_name), height='70px')
        
        for results_list, label_text in ((image_results['all_images'], 
                                     f'All images: {'Correct' if is_all_correct else 'Wrong'}'),
                                    (image_results['base_images'],
                                     f'Base images: {'Correct' if is_base_correct else 'Wrong'}'),
                                    (render_only_results,
                                     f'Render images: {'Correct' if is_render_correct else 'Wrong'}')):
            section_list.append(html_maker.LabelView(label_text))
            section_list.append(build_results_view(
                image_name, dir_dict, results_list, main_pca_view))
        
        column_list.append(html_maker.SectionView(section_list))
    column_list = [html_maker.LabelView(f'{title}: Percision {100 * num_all_correct / len(json_data)}% (All),'
                                        f' {100 * num_base_correct / len(json_data)}% (Base),'
                                        f' {100 * num_render_correct / len(json_data)}% (All Rendered)', size='24px')] \
                   + column_list

    data=html_maker.ColumnView(column_list)
    return data

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
    
def build_results_view(image_name, dir_dict, results_list, main_pca_view):
    res_list = []
    for res in results_list[:4]:
        col_list = [
            html_maker.LabelView(f'{res['image']}', size='14px'),
            html_maker.ImageView(
                find_existing_image_path(dir_dict['images'], res['image'])),
            html_maker.LabelView(f'Loss: {res['loss']: .5}', size='14px'),
            html_maker.LabelView('PCA:'),
            html_maker.RowView([
                html_maker.ImageView(
                    find_existing_image_path(dir_dict['pca'], res['image']), height='70px'),
                main_pca_view
            ])
        ]
        if 'loss2D' in dir_dict:
            col_list.append(html_maker.LabelView('Loss2D:'))
            col_list.append(html_maker.ImageView(
                    find_existing_image_path(dir_dict['loss2D'], f'{image_name}_{res['image']}'), height='70px'))
        res_list.append(html_maker.SectionView(col_list))
    view = html_maker.RowView(res_list)
    return view

if __name__ == '__main__':
    algorithm = 'dino_denoise_croped'
    dir_dict = {
        'images': '/root/feature_matching/croped_images',
        'pca': f'/root/feature_matching/{algorithm}_pca',
        'loss2D': f'/root/feature_matching/{algorithm}_loss2D'
    }
    json_file = f'/root/feature_matching/denoised_features/vit_base_patch14_dinov2.lvd142m_s14/results.json'
    output_html = f'new_dino_denoise_results.html'
    data_to_present = arrange_results(json_file, dir_dict, 'Dino - denoised')
    html = html_maker.generate_html(data_to_present)

    with open(output_html, 'w') as f:
        f.write(html)