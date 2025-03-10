# Configuration

## Install requirements
```bash
python3 -m venv venv
source venv/bin/activate
```
(Optional) With Homebrew installed Python 3.11+
```bash
pip config set global.break-system-packages true
```
run `pip install -r requirements.txt` to install all the required packages.

## Download Checkpoints
You need to search online and download `groundingdino_swint_ogc.pth` to the `src/checkpoints` folder.

Also, when you run project,there will be a model automatically downloaded to a directory in your home folder, which is by default: 
[/Users/YOUR_USERNAME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth](/Users/YOUR_USERNAME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth)

## Set up Database Connection
   Add a `.env` file to your project and store the database connection details in it. The `.env` file should look like this:
   ```plaintext
    DB_HOST=your_database_host
    DB_PORT=your_database_port
    DB_NAME=your_database_name
    DB_USER=your_database_user
    DB_PASSWORD=your_database_password
   ```
## Front-end Repo
[https://github.com/goldenSTAME/similarity-detection](https://github.com/goldenSTAME/similarity-detection)

# Steps to Run the Code

To start the backend server, run the following commands in your terminal:

```bash
python -m src.run
```