from setuptools import setup, find_packages

setup(name="diurnal_misinformation", 
      version="0.1",
      pacakges = find_packages(
          exclude=['diurnal_misinformation.local']
      ))