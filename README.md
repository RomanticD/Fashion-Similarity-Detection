# Image Similarity Detection Backend

A Flask-based backend system for detecting and comparing image similarities using feature vectors and efficient nearest neighbor search.

## Configuration

### Install requirements
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# (Optional) With Homebrew installed Python 3.11+
pip config set global.break-system-packages true

# Install dependencies
pip install -r requirements.txt
```

### Download Checkpoints
You need to search online and download `groundingdino_swint_ogc.pth` to the `src/checkpoints` folder.

Also, when you run the project, there will be a model automatically downloaded to a directory in your home folder, which is by default: 
`/Users/YOUR_USERNAME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`

### Set up Database Connection
Add a `.env` file to your project and store the database connection details in it. The `.env` file should look like this:
```plaintext
DB_HOST=your_database_host
DB_PORT=your_database_port
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
```

### Vector Indexing
This project uses scikit-learn's nearest neighbors implementation to create an efficient vector index for fast image similarity search. Before running the server, you should build the initial vector index:

```bash
# Build the vector index from existing database records
python -m src.utils.build_index
```

The index will be automatically saved to the project root directory as:
- `vector_nn_index.pkl`: Contains the nearest neighbors model and normalized vectors
- `vector_id_map.json`: Maps index positions to database IDs

The index is used by the `/relay_image` endpoint to quickly find similar images without scanning the entire database. The system will automatically rebuild the index if it's missing or corrupted.

## Steps to Run the Project

### 1. Initial Setup
```bash
# Clone the repository
git clone https://github.com/RomanticD/Fashion-Similarity-Detection
cd Fashion-Similarity-Detection

# Set up virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download required checkpoint files
# (See "Download Checkpoints" section)

# Create .env file with database credentials
# (See "Set up Database Connection" section)
```

### 2. Process Images with GroundingDINO
```bash
# First, process and upload images using GroundingDINO
python -m src.db.uploads.batch_upload
```

This step will:
- Use GroundingDINO to detect and segment clothing items in images
- Extract feature vectors from segmented images
- Upload the segmented images and their feature vectors to the database

### 3. Build Vector Index
```bash
# Build the vector index for fast similarity search
python -m src.utils.build_index
```

### 4. Start the Backend Server
```bash
# Start the Flask development server
python -m src.run
```

The server will start running on `http://0.0.0.0:5001`.

## API Endpoints

### Image Similarity Search
```
POST /relay_image
```

**Request Body:**
```json
{
  "image_base64": "base64_encoded_image_data",
  "num": 5
}
```

**Response:**
```json
[
  {
    "id": 123,
    "similarity": 0.95,
    "processed_image_base64": "base64_encoded_image_data"
  },
  ...
]
```

## Database Schema

The core table for image similarity is:

```sql
CREATE TABLE splitted_images (
    id                  int auto_increment primary key,
    splitted_image_id   varchar(255) not null,
    splitted_image_path varchar(255) not null,
    original_image_id   varchar(255) not null,
    bounding_box        text null,
    create_time         datetime default CURRENT_TIMESTAMP null,
    update_time         datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP,
    vector              json null,
    splitted_image_data longblob null,
    constraint splitted_image_id unique (splitted_image_id)
);

CREATE INDEX fk_original_image ON splitted_images (original_image_id);
```

## Front-end Repository
[https://github.com/goldenSTAME/similarity-detection](https://github.com/goldenSTAME/similarity-detection)