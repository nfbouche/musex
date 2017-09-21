from setuptools import setup, find_packages

setup(
    name='musex',
    version='0.1',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['numpy', 'matplotlib', 'astropy', 'mpdaf', 'aplpy',
                      'lineid_plot', 'joblib', 'dataset', 'scipy'],
    # scripts=[],
    # entry_points={
    #     'console_scripts': [
    #         'fix-icrs = muse_analysis.scripts.fix_icrs:main'
    #     ],
    # },
)
