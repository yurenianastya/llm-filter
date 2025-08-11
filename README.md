
# LLM Filter

This project is focused on creating custom Large Language Model (LLM) filters using FastAPI, RabbitMQ, and Ollama. It utilizes pre-trained text classification models from Hugging Face to filter and process incoming messages. The goal is to filter messages based on their content and provide the model's output if the message passes the filtering process.

## API Reference

#### Send prompt

`POST /prompt`

**Description**: This endpoint sends a text message to a pre-processing filter service. If the filtering result is positive, it also returns the model's output.

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `message` | `string` | **Required**. The text message to be processed. |

**Response**: 
- If the message passes the filter, the model's output is returned alongside the filtering result.
- If the message fails the filter, an error message is returned.

## Environment Variables

To run this project, you will need to add the following environment variables to your `.env` file:

- `OLLAMA_MODELS_PATH`: Path to your locally stored Ollama models.
- `OLLAMA_HOST`: The URL for the Ollama API server.
- `OLLAMA_MODEL`: The specific Ollama model to use (e.g., `tinyllama:latest`).
- `HF_HUB`: Directory where Hugging Face models are cached.
- `HF_MODEL`: The Hugging Face model to be used for filtering.
- `HF_TOKEN`: Your Hugging Face API token (if required for accessing models).
- `DATASET`: Path to locally installed dataset (Jigsaw).

### Run Locally

1. **Download the Ollama model**:
   - Download the Ollama model to your local machine.
   - Update the `.env` file with the path to the downloaded model.
   - Example path: `/usr/share/ollama/.ollama/models`
   - Currently using: `tinyllama:latest`
   
2. **Download the Hugging Face transformer model**:
   - Visit [Hugging Face](https://huggingface.co/) and download the model you want to use for filtering.
   - Update the `.env` file with the path to the model.
   - Example path: `/.cache/huggingface/hub`
   - Currently using: `Intel/toxic-prompt-roberta`

3. **Download Kaggle dataset**:
   - Visit the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).
   - Click Download All to get the dataset ZIP archive.
   - Extract the archive to your local project directory.
   - Create a local folder structure for example: `./filter/datasets/`
   - Move `train.csv` into that folder.
   - Ensure your code refers to correct location in `load_dataset().`

4. **Launch Docker Compose**:
   - Run `docker-compose up` to start the application.
   - The initial download of models and dependencies might take a while. Please be patient as the libraries and models are being fetched.

**Note**: Ensure that Docker is installed and running on your machine before launching the application.

## Screenshots

Here are examples of filtering results:

**Positive Result**:
The message passed the filter and returned a positive result.

[![Positive Result](https://i.postimg.cc/prXz2ZC7/image.png)](https://postimg.cc/PLpj4Dnw)

**Negative Result**:
The message failed the filter and returned a negative result.

[![Negative Result](https://i.postimg.cc/9XJyL6Fj/image.png)](https://postimg.cc/r0GF1459)

## Roadmap

### Current Features
- **Preprocessing Filtering**: Initial version of the filter which processes incoming messages using predefined criteria before passing them to the model. This step ensures only relevant messages are processed further.
- **Postprocessing Filtering**: Added postprocessing functionality that allows further filtering and validation of the model's output.
- _In development_ **Semantic filtering**: Currently testing functionality of comparing semantic meanings of text messages alongside classification filtering