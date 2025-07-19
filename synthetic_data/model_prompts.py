def new_instruction_prompt(task_type:str, pool_text:str) -> str:
    """
    Simple prompt generator depending on the step in self-instruct pipeline
    Args:
        human_made_task: str
        task_type: str
    Returns:
        str: prompt for the model
    """
    if task_type == "new_task":
        return f"""
        Come up with a series of tasks: {pool_text} \n Task:
        """
    elif task_type == "instruction_classification":
        # TODO: seperate pool text based on "Task:"
        return f"""
        Can the following task be regarded as a classification task with finite output labels? \n {pool_text}
        """
    elif task_type == "input_first_instance_generation":
        # TODO:
        return f"""
        Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly. \n {pool_text}
        """
    elif task_type == "output_first_instance_generation":
        return f"Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, 
        just generate the correct class label.\n {pool_text}"
    else:
        raise ValueError(f"Invalid task type: {task_type}")
