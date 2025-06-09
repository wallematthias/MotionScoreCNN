from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

def load_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

common_requirements = load_requirements('requirements.txt')
mac_requirements = load_requirements('requirements-mac.txt')
unix_requirements = load_requirements('requirements-unix.txt')

setup(
    name='motionscore',
    version='1.0.0',
    author='Matthias Walle',
    author_email='matthias.walle@ucalgary.ca',
    description='Motionscoring for HR-pQCT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/OpenMSKImaging/MotionScoreCNN',
    packages=find_packages(),
    install_requires=common_requirements,
    extras_require={
        "mac": mac_requirements,
        "unix": unix_requirements
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'motionscore=motionscore.motionscore:main',
        ],
    },
)
