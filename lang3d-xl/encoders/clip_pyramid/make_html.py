import json
from pathlib import Path

# Load the JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Create HTML content for a class
def create_html_content(class_name, datas, folders_names, names, im_width=80):
    # Create a dictionary for data2 based on the stem of the file paths
    datas_by_stem = [{Path(k).name: (v, k) for k, v in data.items()} for data in datas]
    sorted_data1 = sorted(datas[0].items(), key=lambda item: item[1], reverse=False)
    
    mean_scores = [0] * len(datas)
    for folder_num, data in enumerate(datas):
        num_of_images = 0
        for _, score in data.items():
            if score == score:
                mean_scores[folder_num] += score
                num_of_images += 1
        mean_scores[folder_num] /= num_of_images

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{class_name}</title>
    <style>
        body {{
            background-color: #202020;
            color: #e0e0e0;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }}
        .score {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .image {{
            display: block;
            margin: 0 auto;
        }}
    </style>
</head>
<body>
    <h1>{class_name}</h1>
    <table>
        <thead>
            <tr>
                <th>GT</th>
"""

    for score, name in zip(mean_scores, names):
        html_content += f"""<th>{name} (AP: {score * 100:.2f}%)</th>"""
                
    html_content += """
            </tr>
        </thead>
        <tbody>
"""

    for image1, score1 in sorted_data1:

        html_content += f"""            <tr>
                <td>
                    <div class="score">{str(Path(image1).stem)}: {100:.2f}%</div>
                    <img src="{str(Path('HolyScenes_vis/cathedral/st_paul') / Path(image1).relative_to(Path(image1).parents[1])).replace('.jpg', '_vis.jpg')}" alt="{class_name}" class="image" width="{im_width}">
                </td>
                <td>
                    <div class="score">{str(Path(image1).stem)}: {score1 * 100:.2f}%</div>
                    <img src="{image1.replace('.jpg', '_vis.jpg')}" alt="{class_name}" class="image" width="{im_width}">
                </td>
"""
        for data_by_stem in datas_by_stem[1:]:
            score, image = data_by_stem.get(Path(image1).name, 'N/A')
            html_content += f"""                <td>
                    <div class="score">{str(Path(image1).stem)}: {score * 100:.2f}%</div>
                    <img src="{image.replace('.jpg', '_vis.jpg')}" alt="{class_name}" class="image" width="{im_width}">
                </td>
"""
        html_content += f"""            </tr>"""

    html_content += """        </tbody>
    </table>
</body>
</html>"""

    return html_content

# Main process
def generate_htmls(json_paths, names):
    # Extract folder names
    folders_names = [Path(json_path).parent.name for json_path in json_paths]

    # Load JSON data
    datas = [load_json(json_path) for json_path in json_paths]

    # Iterate over classes
    for class_name in datas[0].keys():
        class_datas = [data.get(class_name, {}) for data in datas] # Use empty dict if class not in data

        # Create HTML content for the class
        html_content = create_html_content(class_name, class_datas, folders_names, names)

        # Save the HTML file
        output_path = Path(f'{class_name}.html')
        with open(str(Path(json_paths[0]).parents[2] / output_path), 'w') as html_file:
            html_file.write(html_content)

# Example usage
# json_paths = ['/root/feature-3dgs/data/st_paul/vis_embeddings_mean/data.json',
#               '/root/feature-3dgs/data/st_paul/vis_embeddings_mean_normalized/data.json',
#               '/root/feature-3dgs/data/st_paul/vis_3d_embeddings_mean_normalized/data.json',
#               '/root/feature-3dgs/data/st_paul/clipseg_ft/data.json']
json_paths = ['/root/feature-3dgs/data/st_paul/vis_embeddings_mean_normalized/data.json']
names = ['CLIP_ft Notmalized']
for num in range(1,8):
    json_paths.append(f'/root/feature-3dgs/data/st_paul/vis_embeddings_scale_{num}/data.json')
    names.append(f'Scale {num}')

generate_htmls(json_paths, names)