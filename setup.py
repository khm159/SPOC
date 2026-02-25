from setuptools import setup, find_packages

setup(
    name='SPOC',
    version='0.0.1',
    description='SPOC: Safety-aware planning under partial observability and physical constraints',
    author='Hyumgmin Kim',
    packages=find_packages(
        include=[
            'SPOC', 
            'SPOC.*'
        ]
    )
)