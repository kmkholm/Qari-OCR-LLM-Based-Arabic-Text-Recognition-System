#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qari-OCR GUI Application
LLM-Based Arabic Text Recognition System with Graphical Interface

Author: Dr. Mohammed Tawfik
Email: kmkhol01@gmail.com
Version: 2.0 - Fixed Edition
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import traceback
from pathlib import Path
import time

# Import required libraries for model processing - MODULE LEVEL IMPORTS
try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from pdf2image import convert_from_path
    from PIL import Image
    import arabic_reshaper
    from bidi.algorithm import get_display
    from fpdf import FPDF
    LIBRARIES_IMPORTED = True
    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"Warning: Some libraries not available: {e}")
    LIBRARIES_IMPORTED = False

# GUI Application Class
class QariOCRGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Qari-OCR: Arabic Text Recognition System - Dr. Mohammed Tawfik (Fixed Edition)")
        self.root.geometry("800x750")
        self.root.resizable(True, True)
        
        # Variables (non-tkinter)
        self.model_loaded = False
        self.processing = False
        self.selected_files = []
        self.output_folder = os.getcwd()
        
        # Initialize tkinter variables as None (will be set in setup_gui)
        self.output_folder_var = None
        self.hf_token = None
        self.wandb_key = None
        self.model_cache_dir = None
        self.use_local_model = None
        self.local_model_path = None
        
        # Author Information
        self.author_name = "Dr. Mohammed Tawfik"
        self.author_email = "kmkhol01@gmail.com"
        
        # Setup GUI
        self.setup_gui()
        
    def create_menu(self):
        """Create menu bar with About option"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_separator()
        help_menu.add_command(label="Contact Developer", command=self.show_contact)
        
    def show_about(self):
        """Show About dialog"""
        about_text = f"""Qari-OCR: Arabic Text Recognition System
        
LLM-Based OCR solution using Qwen2VL Vision-Language Model

Author: {self.author_name}
Email: {self.author_email}

Features:
• Arabic text recognition from PDFs and images
• Real-time progress tracking
• User-friendly GUI interface
• High accuracy using state-of-the-art AI
• Fixed import and cache directory issues

System Requirements:
• Hugging Face Token (required)
• WandB API Key (optional)
• Python 3.7+ with tkinter

Version: 2.0 - Fixed Edition
License: MIT

Thank you for using Qari-OCR!"""
        messagebox.showinfo("About Qari-OCR", about_text)
        
    def show_contact(self):
        """Show contact information"""
        contact_text = f"""Contact Information:

Developer: {self.author_name}
Email: {self.author_email}

For support, bug reports, or feature requests,
please contact the developer via email.

Thank you for using Qari-OCR!"""
        messagebox.showinfo("Contact Developer", contact_text)
        
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Initialize tkinter variables (after root window is created)
        self.output_folder_var = tk.StringVar(value=self.output_folder)
        self.hf_token = tk.StringVar()
        self.wandb_key = tk.StringVar()
        self.model_cache_dir = tk.StringVar(value=os.path.expanduser("~/.cache/huggingface"))
        self.use_local_model = tk.BooleanVar(value=False)
        self.local_model_path = tk.StringVar()
        
        # Setup cache directory trace
        self.model_cache_dir.trace('w', self.on_cache_dir_changed)
        
        # Create menu
        self.create_menu()
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure root grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Configuration Section
        config_frame = ttk.LabelFrame(main_frame, text="API Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        # Hugging Face Token
        ttk.Label(config_frame, text="Hugging Face Token:").grid(row=0, column=0, sticky=tk.W, pady=2)
        hf_entry = ttk.Entry(config_frame, textvariable=self.hf_token, show="*", width=60)
        hf_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # WandB API Key (optional)
        ttk.Label(config_frame, text="WandB API Key:").grid(row=1, column=0, sticky=tk.W, pady=2)
        wandb_entry = ttk.Entry(config_frame, textvariable=self.wandb_key, show="*", width=60)
        wandb_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        
        # Load Model Button
        load_btn = ttk.Button(config_frame, text="Load Model", command=self.load_model_threaded)
        load_btn.grid(row=0, column=2, rowspan=2, padx=(20, 0), pady=2)
        
        # Model Status
        self.model_status = ttk.Label(config_frame, text="Enter API keys above, then click 'Load Model' for model management options", foreground="blue")
        self.model_status.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Model Management Section (initially hidden)
        self.model_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="10")
        self.model_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        self.model_frame.columnconfigure(1, weight=1)
        self.model_frame.grid_remove()  # Initially hide this section
        
        # Cache directory info label - MOVED HERE AFTER model_frame creation
        self.cache_info_label = ttk.Label(self.model_frame, text="Model will be saved to the selected cache directory", foreground="green")
        self.cache_info_label.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(5, 2))
        
        # Use local model checkbox
        local_model_check = ttk.Checkbutton(self.model_frame, text="Use Local Model", 
                                          variable=self.use_local_model,
                                          command=self.toggle_local_model)
        local_model_check.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Cache directory
        ttk.Label(self.model_frame, text="Cache Directory:").grid(row=2, column=0, sticky=tk.W, pady=2)
        cache_entry = ttk.Entry(self.model_frame, textvariable=self.model_cache_dir, width=50)
        cache_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=2)
        ttk.Button(self.model_frame, text="Browse", 
                  command=lambda: self.browse_cache_directory()).grid(row=2, column=2, padx=(0, 5), pady=2)
        
        # Local model path
        ttk.Label(self.model_frame, text="Local Model Path:").grid(row=3, column=0, sticky=tk.W, pady=2)
        local_path_entry = ttk.Entry(self.model_frame, textvariable=self.local_model_path, width=50, state="disabled")
        local_path_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=2)
        ttk.Button(self.model_frame, text="Browse", 
                  command=lambda: self.browse_folder(self.local_model_path, "Select Local Model Directory"), 
                  state="disabled").grid(row=3, column=2, padx=(0, 5), pady=2)
        
        # File Selection Section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        # File list
        self.file_listbox = tk.Listbox(file_frame, height=6)
        self.file_listbox.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # File selection buttons
        btn_frame = ttk.Frame(file_frame)
        btn_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        ttk.Button(btn_frame, text="Select Files", command=self.select_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Select Folder", command=self.select_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear List", command=self.clear_files).pack(side=tk.LEFT, padx=(0, 5))
        
        # Output Section
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        output_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_folder_var, width=50)
        output_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=2)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_folder).grid(row=0, column=2, padx=(0, 5), pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.process_btn = ttk.Button(control_frame, text="Start OCR Processing", command=self.start_processing, state="disabled")
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="View Results", command=self.view_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Export PDF", command=self.export_pdf).pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Log Section
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main frame row weights
        main_frame.rowconfigure(7, weight=1)
        
    def browse_cache_directory(self):
        """Browse for cache directory"""
        directory = filedialog.askdirectory(initialdir=self.model_cache_dir.get())
        if directory:
            self.model_cache_dir.set(directory)
            
    def browse_folder(self, var, title):
        """Browse for folder"""
        directory = filedialog.askdirectory(title=title)
        if directory:
            var.set(directory)
            
    def browse_output_folder(self):
        """Browse for output folder"""
        directory = filedialog.askdirectory(initialdir=self.output_folder_var.get())
        if directory:
            self.output_folder_var.set(directory)
            self.output_folder = directory
            
    def on_cache_dir_changed(self, *args):
        """Called when cache directory changes"""
        cache_dir = self.model_cache_dir.get()
        if os.path.exists(cache_dir):
            try:
                size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                          for f in os.listdir(cache_dir) 
                          if os.path.isfile(os.path.join(cache_dir, f)))
                size_mb = size / (1024 * 1024)
                self.cache_info_label.config(text=f"Cache directory: {cache_dir} (Size: {size_mb:.1f} MB)", foreground="green")
            except:
                self.cache_info_label.config(text=f"Cache directory: {cache_dir}", foreground="green")
        else:
            self.cache_info_label.config(text=f"Cache directory: {cache_dir} (will be created)", foreground="orange")
            
    def toggle_local_model(self):
        """Toggle local model usage"""
        state = "normal" if self.use_local_model.get() else "disabled"
        # Find the local path entry and browse button
        for widget in self.model_frame.winfo_children():
            if isinstance(widget, ttk.Entry) and widget.cget("textvariable") == str(self.local_model_path):
                widget.config(state=state)
            elif isinstance(widget, ttk.Button) and "Browse" in widget.cget("text"):
                widget.config(state=state)
                
    def load_model_threaded(self):
        """Load model in a separate thread"""
        if not LIBRARIES_IMPORTED:
            messagebox.showerror("Error", "Required libraries not available. Please install dependencies.")
            return
            
        if not self.hf_token.get().strip():
            messagebox.showerror("Error", "Please enter your Hugging Face token.")
            return
            
        # Start loading in thread
        thread = threading.Thread(target=self.load_model, daemon=True)
        thread.start()
        
    def load_model(self):
        """Load the Qwen2VL model"""
        try:
            self.log_message("🔄 Starting model loading process...")
            self.model_status.config(text="Loading model...", foreground="blue")
            self.root.update()
            
            # Set cache directory
            cache_dir = self.model_cache_dir.get()
            os.environ["HF_HOME"] = cache_dir
            
            self.log_message(f"📁 Cache directory set to: {cache_dir}")
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load model and processor
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            
            self.log_message(f"📥 Loading model from Hugging Face to cache: {cache_dir}")
            
            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                token=self.hf_token.get(),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.log_message("✅ Model loaded successfully")
            
            # Load processor
            self.log_message("🔄 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                token=self.hf_token.get()
            )
            
            self.log_message("✅ Processor loaded successfully")
            
            # Show model management options
            self.model_frame.grid()
            self.model_status.config(text="✅ Model loaded successfully! Model management options are now available.", foreground="green")
            self.process_btn.config(state="normal")
            self.model_loaded = True
            
            self.log_message("🎉 Model loading completed successfully!")
            
        except Exception as e:
            error_msg = f"❌ Error loading model: {str(e)}"
            self.log_message(error_msg)
            self.model_status.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n\n{str(e)}")
            
    def select_files(self):
        """Select files for processing"""
        file_types = [
            ("All supported", "*.pdf *.png *.jpg *.jpeg *.tiff *.bmp *.gif"),
            ("PDF files", "*.pdf"),
            ("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp *.gif"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select files for OCR processing",
            filetypes=file_types
        )
        
        for file_path in files:
            if file_path not in self.selected_files:
                self.selected_files.append(file_path)
                self.file_listbox.insert(tk.END, os.path.basename(file_path))
                
        self.log_message(f"📁 Selected {len(files)} files for processing")
        
    def select_folder(self):
        """Select folder containing files"""
        folder = filedialog.askdirectory(title="Select folder containing files")
        if folder:
            # Find supported files in folder
            supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
            found_files = []
            
            for file_path in Path(folder).rglob('*'):
                if file_path.suffix.lower() in supported_extensions:
                    found_files.append(str(file_path))
                    
            for file_path in found_files:
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)
                    self.file_listbox.insert(tk.END, os.path.basename(file_path))
                    
            self.log_message(f"📁 Found {len(found_files)} supported files in folder")
            
    def clear_files(self):
        """Clear selected files"""
        self.selected_files.clear()
        self.file_listbox.delete(0, tk.END)
        self.log_message("🗑️ Cleared file list")
        
    def start_processing(self):
        """Start OCR processing"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load the model first.")
            return
            
        if not self.selected_files:
            messagebox.showerror("Error", "Please select files to process.")
            return
            
        if self.processing:
            messagebox.showerror("Error", "Processing already in progress.")
            return
            
        # Start processing in thread
        thread = threading.Thread(target=self.process_files, daemon=True)
        thread.start()
        
    def process_files(self):
        """Process selected files"""
        try:
            self.processing = True
            self.process_btn.config(state="disabled")
            
            total_files = len(self.selected_files)
            self.progress_bar.config(maximum=total_files)
            
            self.log_message(f"🚀 Starting OCR processing for {total_files} files...")
            
            for i, file_path in enumerate(self.selected_files):
                self.progress_var.set(f"Processing: {os.path.basename(file_path)} ({i+1}/{total_files})")
                self.progress_bar.config(value=i+1)
                self.root.update()
                
                try:
                    if file_path.lower().endswith('.pdf'):
                        result = self.qari_ocr_pdf(file_path)
                    else:
                        result = self.qari_ocr_image(file_path)
                        
                    self.log_message(f"✅ Completed: {os.path.basename(file_path)}")
                    
                except Exception as e:
                    error_msg = f"❌ Error processing {os.path.basename(file_path)}: {str(e)}"
                    self.log_message(error_msg)
                    self.log_message(f"Traceback: {traceback.format_exc()}")
                    
            self.progress_var.set("🎉 Processing completed!")
            self.log_message("🎉 All files processed successfully!")
            
        except Exception as e:
            error_msg = f"❌ Processing error: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Processing Error", error_msg)
            
        finally:
            self.processing = False
            self.process_btn.config(state="normal")
            
    def qari_ocr_pdf(self, pdf_path):
        """Process PDF file"""
        self.log_message(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        all_text = []
        for i, image in enumerate(images):
            self.log_message(f"  📖 Processing page {i+1}/{len(images)}")
            
            # Save temporary image
            temp_image_path = f"temp_page_{i+1}.png"
            image.save(temp_image_path, "PNG")
            
            try:
                # Process with OCR
                text = self.qari_ocr_image(temp_image_path)
                all_text.append(f"--- Page {i+1} ---\n{text}\n")
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    
        # Combine all text
        combined_text = "\n".join(all_text)
        
        # Save results
        self.save_results(pdf_path, combined_text)
        
        return combined_text
        
    def qari_ocr_image(self, image_path):
        """Process image file"""
        self.log_message(f"🖼️ Processing image: {os.path.basename(image_path)}")
        
        # Prepare the message for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Extract all Arabic text from this image. Return only the Arabic text without any explanations or additional commentary."}
                ]
            }
        ]
        
        # Process the image
        try:
            # Use process_vision_info to handle the images properly
            if LIBRARIES_IMPORTED:
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Prepare inputs
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # Move inputs to the same device as the model
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to("cpu")
                
                # Generate response
                with torch.no_grad():  # This should work now with module-level torch import
                    outputs = self.model.generate(**inputs, max_new_tokens=1000)
                
                # Decode the response
                generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                # Extract the Arabic text
                response = generated_texts[0]
                if "Extract all Arabic text from this image" in response:
                    # Find the actual Arabic text after the prompt
                    arabic_start = response.find("Extract all Arabic text from this image") + len("Extract all Arabic text from this image")
                    arabic_text = response[arabic_start:].strip()
                else:
                    arabic_text = response
                    
                # Clean up the Arabic text
                arabic_text = self.clean_arabic_text(arabic_text)
                
                # Save results
                self.save_results(image_path, arabic_text)
                
                return arabic_text
            else:
                raise Exception("Required libraries not available")
                
        except Exception as e:
            self.log_message(f"ERROR processing {os.path.basename(image_path)}: {str(e)}")
            self.log_message(f"ERROR in process_vision_info: {str(e)}")
            raise
            
    def clean_arabic_text(self, text):
        """Clean and format Arabic text"""
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Reshape Arabic text for proper display
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            
            return bidi_text
            
        except Exception as e:
            self.log_message(f"Warning: Could not clean Arabic text: {str(e)}")
            return text
            
    def save_results(self, file_path, text):
        """Save OCR results to file"""
        try:
            # Create output filename
            base_name = Path(file_path).stem
            output_path = os.path.join(self.output_folder, f"{base_name}_ocr.txt")
            
            # Save to text file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"OCR Results for: {os.path.basename(file_path)}\n")
                f.write(f"Processed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                f.write(text)
                
            self.log_message(f"💾 Results saved to: {output_path}")
            
        except Exception as e:
            self.log_message(f"❌ Error saving results: {str(e)}")
            
    def view_results(self):
        """View OCR results"""
        if not os.path.exists(self.output_folder):
            messagebox.showerror("Error", "Output folder does not exist.")
            return
            
        # Find result files
        result_files = []
        for file in os.listdir(self.output_folder):
            if file.endswith('_ocr.txt'):
                result_files.append(os.path.join(self.output_folder, file))
                
        if not result_files:
            messagebox.showinfo("No Results", "No OCR result files found.")
            return
            
        # Show results dialog
        self.show_results_dialog(result_files)
        
    def show_results_dialog(self, result_files):
        """Show results in a dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("OCR Results")
        dialog.geometry("600x400")
        
        # Create text widget
        text_widget = scrolledtext.ScrolledText(dialog, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load content from first result file
        if result_files:
            try:
                with open(result_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_widget.insert(tk.END, content)
            except Exception as e:
                text_widget.insert(tk.END, f"Error reading file: {str(e)}")
                
        # Make text read-only
        text_widget.config(state=tk.DISABLED)
        
    def export_pdf(self):
        """Export results to PDF"""
        try:
            if not os.path.exists(self.output_folder):
                messagebox.showerror("Error", "Output folder does not exist.")
                return
                
            # Find result files
            result_files = []
            for file in os.listdir(self.output_folder):
                if file.endswith('_ocr.txt'):
                    result_files.append(os.path.join(self.output_folder, file))
                    
            if not result_files:
                messagebox.showinfo("No Results", "No OCR result files found.")
                return
                
            # Create PDF
            pdf_path = os.path.join(self.output_folder, "OCR_Results_Compilation.pdf")
            self.create_pdf_report(result_files, pdf_path)
            
            messagebox.showinfo("Success", f"PDF report created:\n{pdf_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error creating PDF report:\n{str(e)}")
            
    def create_pdf_report(self, result_files, output_path):
        """Create PDF report from result files"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.cell(0, 10, "Qari-OCR Results Report", ln=True, align="C")
        pdf.cell(0, 5, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.cell(0, 10, "", ln=True)  # Empty line
        
        # Process each result file
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Add content to PDF
                pdf.set_font("Arial", size=10)
                lines = content.split('\n')
                
                for line in lines:
                    if len(line) > 100:  # Wrap long lines
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 100:
                                current_line += word + " "
                            else:
                                pdf.cell(0, 5, current_line.strip(), ln=True)
                                current_line = word + " "
                        if current_line:
                            pdf.cell(0, 5, current_line.strip(), ln=True)
                    else:
                        pdf.cell(0, 5, line, ln=True)
                        
                pdf.cell(0, 10, "", ln=True)  # Empty line between files
                
            except Exception as e:
                pdf.cell(0, 5, f"Error reading {result_file}: {str(e)}", ln=True)
                
        pdf.output(output_path)
        
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
        # Also print to console for debugging
        print(log_entry.strip())
        
def main():
    """Main function to run the GUI application"""
    print("🚀 Starting Qari-OCR GUI Application...")
    print("📝 Version: 2.0 - Fixed Edition")
    print("👨‍💻 Author: Dr. Mohammed Tawfik")
    print("📧 Email: kmkhol01@gmail.com")
    print("-" * 50)
    
    # Check if libraries are available
    if not LIBRARIES_IMPORTED:
        print("⚠️  Warning: Some required libraries are not available.")
        print("   The application will start but OCR functionality may be limited.")
        print("   Please install missing dependencies:")
        print("   pip install torch transformers qwen-vl-utils pdf2image")
        print("   pip install pillow arabic-reshaper python-bidi fpdf2")
        print("-" * 50)
    
    # Create and run GUI
    root = tk.Tk()
    app = QariOCRGui(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    print("✅ GUI initialized successfully")
    print("🔧 Fixed Issues:")
    print("   ✅ Module-level imports for torch and process_vision_info")
    print("   ✅ Proper order of GUI component creation")
    print("   ✅ Enhanced cache directory functionality")
    print("   ✅ Improved error handling and logging")
    print("-" * 50)
    print("🎯 Ready to use! Enter your API keys and click 'Load Model'")
    
    # Start the main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Application terminated by user")
    except Exception as e:
        print(f"\n❌ Application error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()