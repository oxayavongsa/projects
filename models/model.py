import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5Chatbot:
    def __init__(self, model_name='t5-small', device=None):
        """
        Initializes the T5 chatbot with the specified model and tokenizer.
        Args:
            model_name: Name of the pre-trained T5 model to use (default: t5-small).
            device: Device to load the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
    
    def generate_response(self, input_text, max_length=50):
        """
        Generates a response for the given input text using the trained T5 model.
        Args:
            input_text: The input dialogue string.
            max_length: Maximum length of the generated response.
        Returns:
            response: Generated response string.
        """
        # Preprocess the input text for the T5 model
        input_ids = self.tokenizer.encode(f"dialogue: {input_text} </s>", return_tensors="pt").to(self.device)
        
        # Generate the response using the T5 model
        output_ids = self.model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
        
        # Decode the generated response to text
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return response

    def load_model_checkpoint(self, checkpoint_path):
        """
        Loads a saved model checkpoint from a specified file path.
        Args:
            checkpoint_path: The path to the saved model checkpoint (.pt file).
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")

    def save_model_checkpoint(self, epoch, optimizer, file_path, loss):
        """
        Saves the model checkpoint to a specified file path.
        Args:
            epoch: Current epoch number.
            optimizer: Optimizer used for training.
            file_path: The path to save the checkpoint.
            loss: Current training loss to be saved with the checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved at epoch {epoch}")
