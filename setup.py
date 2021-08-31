import os
from setuptools import setup

current_path = os.path.abspath('C:\\Users\\Peter\\Desktop\\Personal\\11_Repository\\Call of Duty Related\\Call-Of-Duty-Warzone-Analysis')


def read_file(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return reader.read()

setup(
    name='warzone analysis',
    version='2.3.0',
    # packages=['Classes', 'Utils',],
    packages=['warzone',],
    author='Peter Rigali',
    author_email='peterjrigali@gmail.com',
    license='MIT',
    description='Call of Duty Warzone Analysis',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://call-of-duty-warzone-analysis.readthedocs.io/en/latest/intro.html#write-ups-and-examples',
)
