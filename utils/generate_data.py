from typing import List
from typing_extensions import Annotated
import uuid

import dotenv
from pydantic import BaseModel, Field
from utils.llm import get_aoai_client

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.sdg_utils import (
    get_instruction,
    get_domain,
    get_tone,
    get_question_length,
    get_difficulty,
    get_topic,
    get_language,
    set_is_grounded
)

# Load environment variables from the specified file
dotenv.load_dotenv(".env")

class QuestionSchema(BaseModel):
    question: Annotated[
        List[str],
        Field(..., description="A list of synthetic questions related to the original topic.")
    ]
    response: Annotated[
        List[str],
        Field(..., description="A list of synthetic agent responses corresponding to the synthetic questions.")
    ]
    explanation: Annotated[
        List[str],
        Field(..., description="A list of explanations detailing why the question and agent response are (or are not) grounded based on the provided context.")
    ]


class SyntheticDataGenerator:
    def __init__(self, model):
        self.model = model

        # Load both system messages
        grounded_message_path = "configs/prompts/system_message_grounded_questions.txt"
        not_grounded_message_path = "configs/prompts/system_message_not_grounded_questions.txt"
        
        self.system_message_grounded = get_instruction(grounded_message_path)
        self.system_message_not_grounded = get_instruction(not_grounded_message_path)

    def build_user_prompt(
        self,
        main_chunk: str,
        similar_chunks: list,
        is_grounded: bool,
        domain: str,
        difficulty: str,
        topic: str,
        language: str,
        instructions: str,
        question_length: int,
        task: str,
    ) -> str:
        """Constructs the user prompt with main chunk and similar chunks clearly distinguished."""
        # Format similar chunks
        similar_chunks_text = "\n\n---\n\n".join([
            f"SIMILAR CHUNK {i+1}:\n{chunk}" 
            for i, chunk in enumerate(similar_chunks)
        ])
        
        prompt = (
            f"# MAIN CHUNK:\n{main_chunk}\n\n"
            f"# SIMILAR CHUNKS:\n{similar_chunks_text}\n\n"
            f"# Domain:\n{domain}\n\n"
            f"# Difficulty:\n{difficulty}\n\n"
            f"# Language:\n{language}\n\n"
            f"# Instructions:\n{instructions}\n\n"
            f"# Question Length (number of words):\n{question_length}\n\n"
            f"# Task:\n{task}\n\n"
            f"# Grounded: {is_grounded}\n"
        )
        return prompt
    
    def get_system_message(self, is_grounded: bool) -> str:
        """Select the appropriate system message based on whether the question should be grounded."""
        return self.system_message_grounded if is_grounded else self.system_message_not_grounded

    def invoke_llm(
        self,
        main_chunk: str,
        similar_chunks: list,
        is_grounded: bool,
        domain: str,
        difficulty: str,
        topic: str,
        language: str,
        instructions: str,
        question_length: int,
        task: str,
    ) -> dict:
        """Invokes the LLM using the provided parameters and returns the parsed response."""
        client = get_aoai_client()
        user_prompt = self.build_user_prompt(
            main_chunk, similar_chunks, is_grounded, domain, difficulty, topic,
            language, instructions, question_length, task
        )

        # Select system message based on is_grounded
        system_message = self.get_system_message(is_grounded)

        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                # temperature=0.7,
                max_completion_tokens=5000,
                response_format=QuestionSchema
            )
        except Exception as e:
            print("Error during LLM invocation: %s", e)
            raise

        try:
            if not response.choices:
                raise ValueError("No choices returned from the LLM.")
            parsed = response.choices[0].message.parsed
            result = parsed.model_dump()
            return result
        except Exception as e:
            print("Error parsing LLM response: %s", e)
            raise

    def __call__(
        self,
        main_chunk: str,
        similar_chunks: list,
        is_grounded: bool,
        domain: str,
        difficulty: str,
        topic: str,
        language: str,
        instructions: str,
        question_length: int,
        task: str,
    ) -> dict:
        """Makes the class instance callable. Delegates to invoke_llm."""
        return self.invoke_llm(
            main_chunk=main_chunk,
            similar_chunks=similar_chunks,
            is_grounded=is_grounded,
            domain=domain,
            difficulty=difficulty,
            topic=topic,
            language=language,
            instructions=instructions,
            question_length=question_length,
            task=task,
        )


def generate_single_question(chunk, generator, task_path, all_chunks):
    """Helper function to generate a single question (for parallel execution)"""
    from utils.search import find_similar_chunks
    
    main_chunk = chunk["chunk"]
    domain = get_domain()
    tone = get_tone()
    difficulty = get_difficulty()
    question_length = get_question_length(min_length=4)
    topic = get_topic()
    language = get_language()
    instructions = "None"
    task = get_instruction(task_path)
    is_grounded = set_is_grounded()

    # Find k=5 most similar chunks using k-NN
    similar_chunk_objs = find_similar_chunks(chunk, all_chunks, k=5)
    similar_chunks = [c["chunk"] for c in similar_chunk_objs]
    
    try:
        result = generator(
            main_chunk, similar_chunks, is_grounded, domain,
            difficulty, topic, language,
            instructions, question_length, task
        )
        question = result["question"][0]
        explanation = result["explanation"][0]
        response = result["response"][0]
        unique_id = str(uuid.uuid4())[:8]
        synthetic_chunk_id = f"{chunk['chunk_id']}_synthetic_{unique_id}"
        
        return {
            "success": True,
            "data": {
                "synthetic_question": question,
                "explanation": explanation,
                "synthetic_response": response,
                "chunk_id": chunk['chunk_id'],
                "synthetic_chunk_id": synthetic_chunk_id,
                "is_grounded": is_grounded,
                "main_chunk": main_chunk,
                "similar_chunks": similar_chunks,
                "domain": domain,
                "difficulty": difficulty,
                "tone": tone,
                "language": language,
                "question_length": question_length
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": {
                "error": str(e),
                "chunk_id": chunk["chunk_id"],
                "is_grounded": is_grounded,
                "main_chunk": main_chunk,
                "domain": domain,
                "difficulty": difficulty,
                "tone": tone,
                "language": language,
                "question_length": question_length
            }
        }


def generate_synthetic_questions(chunks, generator, max_workers=10):
    """Generate synthetic questions in parallel using k-NN for similar chunks."""
    all_results = []
    failed_results = []
    task_path = "configs/settings/task_single_grounded_not_grounded_questions.txt"
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunk in chunks:
            future = executor.submit(
                generate_single_question, 
                chunk, 
                generator, 
                task_path,
                chunks  # Pass all chunks for k-NN search
            )
            futures.append(future)
        
        # Calculate total tasks for progress bar
        total_tasks = len(futures)
        
        # Wait for all tasks to complete and collect results
        with tqdm(total=total_tasks, desc="Generating synthetic questions") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    all_results.append(result["data"])
                else:
                    failed_results.append(result["error"])
                pbar.update(1)
    
    return all_results, failed_results





