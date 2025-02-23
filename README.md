# Configuration


## PyTorch Model Download Location

When you run a PyTorch-based project that requires a pre-trained model, PyTorch will automatically download the model file from its online repository the first time you use it. 


### Default Download Location

By default, PyTorch saves the downloaded models to a directory in your home folder, which is: 
[/Users/YOUR_USERNAME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth](/Users/YOUR_USERNAME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth)

### Default Download Location

By default, PyTorch saves the downloaded models to a directory in your home folder, which is:

## Connecting to MySQL Database using PyCharm IDE

### Step-by-Step Guide

#### 1. **Open PyCharm IDE**
   Launch PyCharm IDE and open your project.

#### 2. **Navigate to Database Tool Window**
   On the right-hand side of the PyCharm window, locate the **Database** tool window. If it’s not visible, you can open it by going to **View** > **Tool Windows** > **Database** or using the shortcut `Alt + 1` (Windows/Linux) or `Cmd + 1` (Mac).

#### 3. **Add a New Database Connection**
   - Click the **+** icon located at the top of the **Database** tool window.
   - Select **Data Source** > **MySQL** from the dropdown menu.

#### 4. **Configure the Database Connection**
   Add a .env file to your project and store the database connection details in it. The .env file should look like this:
   
   ```plaintext
    DB_HOST=your_database_host
    DB_PORT=your_database_port
    DB_NAME=your_database_name
    DB_USER=your_database_user
    DB_PASSWORD=your_database_password
