# Fashion Similarity Detection Backend

A Flask-based backend system that detects clothing items in images and finds similar fashion items using feature vectors and efficient nearest neighbor search.

## Features

- **Clothing Detection**: Uses GroundingDINO to automatically detect and segment clothing items in images
- **Feature Extraction**: Extracts feature vectors from detected clothing segments using ResNet50
- **Similarity Search**: Finds visually similar fashion items using vector similarity
- **Efficient Indexing**: Implements nearest neighbor search for fast image retrieval
- **Authentication**: Supports user roles and authentication via Supabase integration
- **Cancellable Operations**: Long-running operations can be tracked and cancelled by clients

## Architecture

The system consists of several key components:

1. **Core ML Components**:
   - `ClothingDetector`: Handles clothing detection in images
   - `ImageSimilarity`: Extracts feature vectors and compares similarities
   - `VectorIndex`: Manages efficient similarity search using nearest neighbors

2. **API Endpoints**:
   - `/upload_image`: Process and upload new fashion images
   - `/relay_image`: Find similar images based on an input image
   - `/image_detail`: Get detailed information about fashion items
   - Authentication endpoints for user management

3. **Database Layer**:
   - Stores processed images, feature vectors, and metadata
   - Maintains relationships between original and segmented images

## Setup and Installation

### Prerequisites

- Python 3.9+
- MySQL database
- Supabase project (for authentication)

### Install Requirements

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# (Optional) With Homebrew installed Python 3.11+
pip config set global.break-system-packages true

# Install dependencies
pip install -r requirements.txt
```

### Download Model Checkpoints

The system requires pretrained models to function properly:

1. Download the GroundingDINO model:
   - Search for and download `groundingdino_swint_ogc.pth`
   - Place it in the `src/checkpoints` folder

2. ResNet50 model:
   - This will be automatically downloaded during first run to:
   - `/Users/YOUR_USERNAME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`

### Set Up Database Connection

Create a `.env` file in the project root with your database credentials:

```
DB_HOST=your_database_host
DB_PORT=your_database_port
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
```

For Supabase authentication, add:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Database Schema

The core table structure for image similarity is:

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

Additionally, you'll need a table for detailed fashion information:

```sql
CREATE TABLE image_details (
    image_id          varchar(255) primary key,
    store_name        varchar(255) null,
    brand             varchar(255) null,
    product_name      varchar(255) null,
    store_description text null,
    url               varchar(255) null,
    rating            float null,
    tags              json null,
    sale_status       varchar(50) null,
    size              json null,
    waist_type        varchar(50) null,
    listing_season    varchar(50) null,
    season            varchar(50) null,
    create_time       datetime default CURRENT_TIMESTAMP null,
    update_time       datetime default CURRENT_TIMESTAMP null on update CURRENT_TIMESTAMP
);
```

## Running the Project

### 1. Initial Data Processing

Before running the server, process some initial images to populate the database:

```bash
# Process and upload images using GroundingDINO
python -m src.db.uploads.batch_upload
```

This will:
- Detect and segment clothing items in images from the `ScreenShots` directory
- Extract feature vectors from segmented images
- Upload the processed data to the database

### 2. Build Vector Index

After uploading images, build the initial vector index for fast similarity search:

```bash
# Build the vector index
python -m src.utils.build_index
```

The index will be saved as:
- `vector_nn_index.pkl`: Contains the nearest neighbors model and normalized vectors
- `vector_id_map.json`: Maps index positions to database IDs

### 3. Start the Backend Server

Launch the Flask development server:

```bash
# Start the server
python -m src.run
```

The server will start running on `http://0.0.0.0:5001`.

## API Endpoints Reference

### Image Upload

```
POST /upload_image
```

**Request Body:**
```json
{
  "image_base64": "base64_encoded_image_data",
  "image_name": "optional_custom_name",
  "force_process": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed and uploaded image, detected 2 clothing segments",
  "request_id": "abc123-uuid",
  "data": {
    "original_image_id": "image_name",
    "segments": [
      {
        "splitted_image_id": "image_name_segment_0",
        "splitted_image_path": "image_name/segment_0.png"
      },
      {
        "splitted_image_id": "image_name_segment_1",
        "splitted_image_path": "image_name/segment_1.png"
      }
    ]
  }
}
```

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
{
  "request_id": "abc123-uuid",
  "results": [
    {
      "id": 123,
      "similarity": 0.95,
      "processed_image_base64": "base64_encoded_image_data"
    },
    ...
  ]
}
```

### Image Details

```
POST /image_detail
```

**Request Body:**
```json
{
  "splitted_image_id": "01_segment_0"
}
```

**Response:**
```json
{
  "success": true,
  "message": {
    "image_id": "01",
    "store_name": "Fashion Store",
    "brand": "Brand Name",
    "product_name": "Product Description",
    "store_description": "Product details...",
    "rating": 4.5,
    "tags": ["casual", "summer", "trendy"],
    "sale_status": "in_stock",
    "size": ["S", "M", "L"],
    "waist_type": "high",
    "listing_season": "2024Spring",
    "season": "summer"
  }
}
```

### Cancel Request

```
POST /cancel_request/:request_id
```

**Response:**
```json
{
  "status": "success",
  "message": "Request abc123-uuid marked for cancellation"
}
```

## Extending the System

### Adding New Detection Models

To add a new clothing detection model:

1. Create a new handler class in `src/core/`
2. Implement the detection interface similar to `ClothingDetector`
3. Register the new model in the appropriate route handler

### Improving Similarity Search

To enhance similarity matching:

1. Experiment with alternative feature extraction models in `src/core/image_similarity.py`
2. Consider implementing the ViT-based extractor in `src/core/image_similarity_vit.py`
3. Adjust the nearest neighbors parameters in `src/core/vector_index.py`

## Testing

The project includes several test scripts in the `test/` directory:

- `test_upload.py`: Test image upload functionality
- `similarity_test.py`: Test similarity search
- `GroundingDINO_test.py`: Test clothing detection

Run tests with:

```bash
python -m test.test_upload
python -m test.similarity_test
python -m test.GroundingDINO_test
```

## Front-end Repository

The companion front-end application is available at:
[https://github.com/goldenSTAME/similarity-detection](https://github.com/goldenSTAME/similarity-detection)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
