import os
import sys

# 获取虚拟环境的 site-packages 路径
venv_site_packages = '/Users/sunyuliang/Desktop/AppBuilder/Python/RD-Test/.venv/lib/python3.12/site-packages'
# 将虚拟环境的 site-packages 路径添加到 Python 搜索路径
sys.path.insert(0, venv_site_packages)

# 原有的配置代码保持不变
project = 'RD'
copyright = '2025, Team2'
author = 'Team2'

# 获取项目根目录（project_root/）
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 将 src 目录添加到 Python 路径
sys.path.insert(0, os.path.join(root_dir, 'src'))

# 调试代码
print("\n✅ 当前 Python 路径:")
for p in sys.path:
    print(f" - {p}")

print("\n✅ 尝试导入 core.groundingdino_handler...")
try:
    from core import groundingdino_handler
    print("✅ 导入成功!")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
