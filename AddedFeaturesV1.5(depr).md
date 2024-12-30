Here's a breakdown of the changes from the documentation to the provided code:

**1. Imports:**

*   **New Import:** `import tiktoken` has been added. This library is used for token counting, which is essential for the new cost estimation feature.

**2. `RAGConfig` Dataclass:**

*   **Modified `chunk_size`:** The default `chunk_size` has been changed from `500` to `600`.
*   **Modified `chunk_overlap`:** The default `chunk_overlap` has been changed from `100` to `150`.
*   **Modified `max_context_chunks`:** The default `max_context_chunks` has been changed from `3` to `12`.

**3. `RAGSystem` Class:**

*   **New Attribute in `__init__`:** A `usage_stats` dictionary has been added to the `__init__` method to track token usage for cost calculation.

    ```python
    self.usage_stats = {
        "embedding_tokens": 0,
        "completion_input_tokens": 0,
        "completion_output_tokens": 0
    }
    ```

*   **New Method: `estimate_processing_costs`:** This method calculates the estimated cost of processing a given content based on token count and OpenAI pricing.

    ```python
    def estimate_processing_costs(self, content: str) -> Dict[str, float]:
        # ... implementation ...
    ```

*   **Modified `load_document` Method:**
    *   This method now includes a check for large documents (more than 25 pages).
    *   For large documents, it calls `estimate_processing_costs` to calculate and display the estimated cost to the user.
    *   It prompts the user for confirmation before proceeding with processing a large document.

    ```python
    def load_document(self, file_path: str) -> Optional[str]:
        # ... (code to load document) ...
        if num_pages > 25:
            # ... (cost estimation and user confirmation) ...
        # ... (rest of the method) ...
    ```

*   **Modified `get_embedding` Method:**
    *   Token usage tracking has been added to this method. It now increments the `embedding_tokens` count in the `usage_stats` dictionary based on the OpenAI API response.

    ```python
    def get_embedding(self, text: str) -> List[float]:
        # ... (embedding generation code) ...
        self.usage_stats["embedding_tokens"] += response.usage.total_tokens
        return embedding
    ```

*   **Modified `generate_answer` Method:**
    *   Token usage tracking for completion calls has been added. It now increments `completion_input_tokens` and `completion_output_tokens` in the `usage_stats` dictionary.

    ```python
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        # ... (prompt generation and OpenAI call) ...
        self.usage_stats["completion_input_tokens"] += response.usage.prompt_tokens
        self.usage_stats["completion_output_tokens"] += response.usage.completion_tokens
        return response.choices[0].message.content
    ```

*   **New Method: `calculate_final_costs`:** This method calculates the final costs based on the accumulated token usage in the `usage_stats` dictionary.

    ```python
    def calculate_final_costs(self) -> Dict[str, float]:
        # ... implementation ...
    ```

**4. `main` Function:**

*   **Modified `main` function:**
    *   The `main` function now includes code to call `rag.calculate_final_costs()` after the interactive query loop.
    *   It prints a detailed "Final Usage and Cost Report" including token counts and calculated costs for embedding and completion.

    ```python
    def main():
        # ... (document loading and query loop) ...
        final_costs = rag.calculate_final_costs()
        print("\nFinal Usage and Cost Report")
        # ... (printing cost details) ...
    ```

**In summary, the key new features and changes are:**

*   **Cost Estimation:** The system now estimates the cost of processing documents, particularly large ones, before proceeding.
*   **Cost Tracking:** The system tracks the actual token usage for both embedding and completion API calls.
*   **Cost Reporting:**  A final report is generated showing the token usage and associated costs.
*   **User Confirmation for Large Documents:** Users are prompted to confirm if they want to process large documents after seeing the estimated cost.
*   **Configuration Changes:** Default values for `chunk_size`, `chunk_overlap`, and `max_context_chunks` have been updated.
*   **`tiktoken` Integration:** The `tiktoken` library is used for token counting.

These changes significantly enhance the practical utility of the RAG system by providing transparency and control over the costs associated with using the OpenAI API.
