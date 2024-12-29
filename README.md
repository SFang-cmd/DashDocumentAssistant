# Sample usage of ColPali Document Processing
To run sample project:
1. Install necessary dependencies:

    a. pdf2image - https://pypi.org/project/pdf2image/
    b. byaldi - https://github.com/AnswerDotAI/byaldi
    c. PIL - 
    d. openai (if using OpenAI multimodal models) - https://platform.openai.com/docs/api-reference/introduction
    e. transformers (HuggingFace)

2. The project runs in 2 overall steps

    a. First, the colpali RAG is called to index files
    b. Next, a multimodal generation model (OpenAI, Claude, etc.) is called to synthesize necessary information and provide feedback to student
    
3. The colpali RAG is assisted by the Byaldi code wrapper (please see above for documentation).