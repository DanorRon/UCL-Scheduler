from setuptools import setup, find_packages

setup(
    name="ucl-scheduler",
    version="1.0.0",
    description="An intelligent rehearsal scheduling system with optimization capabilities",
    author="Ronan Venkat",
    author_email="ronanvenkat@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "ortools>=9.11.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "gspread>=5.0.0",
        "google-auth>=2.0.0",
        "waitress>=3.0.2"
    ],
    extras_require={
        "web": ["flask>=2.3.0"],
    },
    python_requires=">=3.11",
) 