from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirement_file:
    requirements_list = requirement_file.readlines()
    requirements_list = [lib.replace('\n', '') for lib in requirements_list]

requirements = requirements_list

setup(
    name='African-Languages-Speech-Recognition',
    version='0.1.0',
    packages=[''],
    url='https://github.com/African-language-Speech-Recognition/african_language-Speech_Recognition',
    license='MIT License',
    author='10Academy Batch-5 Week-4 Group-4',
    author_email='diyye101@gmail.com',
    description='African languages speech to text transcription',
    install_requires=requirements,
    long_description=readme,
)
