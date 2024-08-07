from setuptools import setup, find_packages

setup(name="diurnal_misinformation", 
      version="0.1",
      packages = find_packages(
          exclude=['diurnal_misinformation.local']
      ),
      install_requires=[
        'diptest',
        'Jinja2',
        'matplotlib',
        'numpy',
        'pandas',
        'pyarrow',
        'scikit-learn',
        'scipy',
        'seaborn',
        'similaritymeasures',
        'suntime',
        'validclust',
        'yellowbrick'
      ])