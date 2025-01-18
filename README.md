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
   In the dialog that appears, enter the following connection details:

   - **Host**: `database-1.c5282akgwxld.ap-southeast-2.rds.amazonaws.com` (Replace with your actual host)
   - **Port**: `3306`
   - **User**: `admin`
   - **Password**: `112345678` 
   - **Database**: `fashion_db`