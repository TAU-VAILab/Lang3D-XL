def generate_html(main_view):
    html_content = """
    <html>
    <head>
        <style>
            /* General dark mode styles */
            body {
                background-color: #121212;
                color: #e0e0e0;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }

            .row {
                display: flex;
                gap: 5px;
                margin-bottom: 20px;
            }

            .column {
                display: flex;
                flex-direction: column;
                gap: 5px;
                margin-bottom: 20px;
            }

            .section {
                border: 2px solid #444;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                background-color: #1e1e1e;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }

            .item {
                margin: 10px;
                padding: 10px;
                border-radius: 5px;
                transition: background-color 0.3s ease;
            }

            .item:hover {
                background-color: #333;
            }

            img {
                height: 200px;
                width: auto;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }

            /* Styling for text items */
            .label {
                font-size: 18px; /* Default size */
                font-weight: bold;
                color: #ffffff;
            }
        </style>
    </head>
    <body>
    """

    html_content += main_view.generate(indent_level=0)

    html_content += """
    </body>
    </html>
    """

    return html_content


class LabelView:
    def __init__(self, data, size='18px'):
        self.data = data
        self.size = size

    def generate(self, indent_level=0):
        indent = ' ' * (indent_level * 4)  # 4 spaces per indent level
        html = ''

        html += f"{indent}<div class='item label' style='font-size: {self.size};'>{self.data}</div>\n"
        
        return html


class ImageView:
    def __init__(self, data: str, height='200px'):
        self.data = data
        self.height = height

    def generate(self, indent_level=0):
        indent = ' ' * (indent_level * 4)  # 4 spaces per indent level
        html = ''

        html += f"{indent}<div class='item'><img src='{self.data}' alt='Image' style='height: {self.height};' /></div>\n"
        
        return html


class RowView:
    def __init__(self, data):
        self.data = data

    def generate(self, indent_level=0):
        indent = ' ' * (indent_level * 4)  # 4 spaces per indent level
        html = ''

        html += f"{indent}<div class='row'>\n"
        for sub_item in self.data:
            html += sub_item.generate(indent_level + 1)
        html += f"{indent}</div>\n"
        
        return html


class ColumnView:
    def __init__(self, data):
        self.data = data

    def generate(self, indent_level=0):
        indent = ' ' * (indent_level * 4)  # 4 spaces per indent level
        html = ''

        html += f"{indent}<div class='column'>\n"
        for sub_item in self.data:
            html += sub_item.generate(indent_level + 1)
        html += f"{indent}</div>\n"
        
        return html


class SectionView:
    def __init__(self, data):
        self.data = data

    def generate(self, indent_level=0):
        indent = ' ' * (indent_level * 4)  # 4 spaces per indent level
        html = ''

        html += f"{indent}<div class='section'>\n"
        for sub_item in self.data:
            html += sub_item.generate(indent_level + 1)
        html += f"{indent}</div>\n"
        
        return html

# Example usage
main_view = ColumnView([
    LabelView('This is a label', size='24px'),  # Custom size
    LabelView('This is another label'),  # Default size
    ImageView(r'images\00291.png'),
    RowView([LabelView('Label in row', size='20px'),
             ImageView(r'images\00291.png'),
             ImageView(r'images\00291.png'),
             ImageView(r'images\00291.png')]),
    ColumnView([LabelView('Label in column', size='20px'),
                ImageView(r'images\00291.png'),
                LabelView('Another label in column', size='20px'),
                ImageView(r'images\00291.png'),
                ImageView(r'images\00291.png')]),
    SectionView([LabelView('Section label', size='28px'),
                ImageView(r'images\00291.png'),
                RowView([LabelView('Nested row label'),
                         ImageView(r'images\00291.png'),
                         ImageView(r'images\00291.png')]),
                ])
    ])

html = generate_html(main_view)

# Write the HTML to a file
with open('output.html', 'w') as f:
    f.write(html)
