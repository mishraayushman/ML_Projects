from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    "this functon return list of requirements"
    require =[]
    with open(file_path) as file_obj:
        require = file_obj.readlines()
        require = [req.replace('\n',"") for req in require]

        if HYPHEN_E_DOT in require:
            require.remove(HYPHEN_E_DOT)
    
    return require

setup(
name="mlproject",
version="0.0.1",
author="Ayushman mishra",
packages=find_packages(),
install_requires=get_requirements("F:\\ML_Projects_new\\requirements.txt")    
)