
# LLM Filter

Project about creating custom LLM filters, using FastAPI, RabbitMQ and Ollama. Currently works by using text classification models from huggingface.
## API Reference

#### Send prompt

```http
  POST /prompt
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `message` | `string` | **Required**. Text mesasge |


Sends `message` to a pre-processing filter service and returns filtering result. If result is positive - also returns model's output.
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file:

`OLLAMA_MODELS_PATH`

`OLLAMA_HOST`

`OLLAMA_MODEL`

`RABBITMQ_HOST`

`HF_HUB`

`HF_MODEL`


## Screenshots

Example of a positive result from the filter

[![image.png](https://i.postimg.cc/N0bjQ8Np/image.png)](https://postimg.cc/PLpj4Dnw)

And a negative one

[![image.png](https://i.postimg.cc/mDJzGNDK/image.png)](https://postimg.cc/r0GF1459)
