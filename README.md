# Qari-OCR: LLM-Based Arabic Text Recognition System
## نظام التعرف على النص العربي قاري

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **LLM-powered OCR solution for Arabic documents using Qwen2VL Vision-Language Model**  
> **حل OCR مدعوم بنماذج اللغة الكبيرة للمستندات العربية باستخدام نموذج Qwen2VL للرؤية واللغة**

---

## ⚠️ IMPORTANT: External API Requirements | مهمة: متطلبات API خارجية

**You MUST provide your own API credentials for this system to work:**
- **Hugging Face Token**: Required to access the Qwen2VL model
- **WandB API Key**: Optional (can be disabled)
- This system uses external LLM APIs, not local processing

**يجب عليك توفير بيانات الاعتماد الخاصة بك لنظام API الخارجي لكي يعمل النظام:**
- **رمز Hugging Face**: مطلوب للوصول لنموذج Qwen2VL
- **رمز WandB API**: اختياري (يمكن تعطيله)
- يستخدم هذا النظام API خارجية للـ LLM، وليس المعالجة المحلية

---

## 📖 About | حول

Qari-OCR is an LLM-based Optical Character Recognition system that leverages the powerful Qwen2VL vision-language model for Arabic text extraction from PDF documents and images. This system uses existing state-of-the-art AI models accessed through external APIs and provides accurate and reliable Arabic text recognition with proper text shaping and bidirectional formatting.

**Key Features | المزايا الرئيسية:**
- 🎯 High accuracy Arabic text recognition using LLM | دقة عالية في التعرف على النص العربي باستخدام نماذج اللغة الكبيرة
- 📄 PDF and multiple image format support | دعم ملفات PDF والصور بتنسيقات متعددة
- 🔤 Proper Arabic text shaping (RTL) | تشكيل النص العربي بشكل صحيح (من اليمين لليسار)
- 📱 Google Colab ready | جاهز للاستخدام في Google Colab
- 🚀 Easy to use interface | واجهة سهلة الاستخدام
- 📤 Export to TXT and formatted PDF | تصدير إلى ملف نصي و PDF منسق
- ⚠️ **Requires External API Access** | **يتطلب الوصول لـ API خارجية**

### How It Works | آلية العمل
This system is **NOT** a standalone OCR engine. It uses:
1. **External LLM API** - Qwen2VL model from Hugging Face
2. **External Services** - Optional WandB for monitoring
3. **Cloud Processing** - Your data is processed through external AI services

---

## 🛠️ Installation | التثبيت

### Prerequisites | المتطلبات
```bash
# System dependencies | متطلبات النظام
apt-get install -y poppler-utils fonts-dejavu-core

# Python packages | حزم Python
pip install transformers>=4.43.0 qwen_vl_utils accelerate>=0.26.0 peft bitsandbytes pdf2image pillow arabic-reshaper python-bidi fpdf2
```

### API Setup Required | إعداد API مطلوب

**⚠️ CRITICAL: You MUST set up your own API access before running!**

1. **Hugging Face Token** | رمز Hugging Face:
   - Go to: https://huggingface.co/settings/tokens
   - Create a new token with read permissions
   - Copy your token (starts with `hf_`)

2. **WandB API Key** (Optional) | رمز WandB API (اختياري):
   - Go to: https://wandb.ai/settings
   - Create API key from settings
   - Can be disabled if not needed

**Required Model Access:**
- **Model**: `NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct`
- **Access**: Your Hugging Face token

---

## 🚀 Quick Start | البدء السريع

> **⚠️ BEFORE YOU START: Make sure you have set up your API keys!**  
> **⚠️ قبل البدء: تأكد من إعداد مفاتيح API الخاصة بك!**

### 1. API Setup First! | إعداد API أولاً!
```python
# Clone or download this repository | انسخ أو حمل هذا المشروع
git clone [your-repo-url]
cd qari-ocr

# Install dependencies | تثبيت المتطلبات
pip install -r requirements.txt
```

### 2. Run in Google Colab | التشغيل في Google Colab
```python
# Simply upload and run the notebook | حمل وشغل الدفتر فقط
# Upload your PDF/image files | حمل ملفات PDF/الصور الخاصة بك
# The system will automatically process them |سيقوم النظام بمعالجتها تلقائياً
```

### 3. Usage | الاستخدام
```python
# Import required libraries | استيراد المكتبات المطلوبة
from qari_ocr import process_documents

# Process your files | معالجة ملفاتك
results = process_documents("path/to/your/files")

# Download results | تحميل النتائج
# - qari_ocr_output.txt
# - qari_ocr_output.pdf
```

---

## 📋 How It Works | آلية العمل

1. **File Upload** | رفع الملفات
   - Supports PDF, PNG, JPG, JPEG, TIF, TIFF, BMP, WEBP | يدعم هذه التنسيقات
   - Automatic format detection | كشف التنسيق التلقائي

2. **Image Processing** | معالجة الصور
   - PDF pages converted to high-resolution images | تحويل صفحات PDF لصور عالية الدقة
   - Image preprocessing for optimal OCR | معالجة مسبقة للصور للحصول على أفضل نتائج

3. **Arabic Text Recognition** | التعرف على النص العربي
   - Uses Qwen2VL vision-language model | يستخدم نموذج Qwen2VL للرؤية واللغة
   - Specialized for Arabic text patterns | متخصص في أنماط النص العربي
   - Maintains original formatting | يحافظ على التنسيق الأصلي

4. **Text Processing** | معالجة النص
   - Arabic text reshaping | إعادة تشكيل النص العربي
   - Bidirectional text algorithm | خوارزمية النص ثنائي الاتجاه
   - Proper RTL rendering | عرض صحيح من اليمين لليسار

5. **Output Generation** | إنتاج المخرجات
   - Plain text file (.txt) | ملف نصي عادي
   - Formatted PDF with Arabic support | ملف PDF منسق بدعم العربية

---

## 📁 File Structure | هيكل المشروع

```
qari-ocr/
├── README.md                 # This file | هذا الملف
├── requirements.txt          # Dependencies | المتطلبات
├── qari_ocr.py              # Main script | السكريبت الرئيسي
├── qari_ocr_colab.ipynb     # Google Colab notebook | دفتر Google Colab
├── examples/                # Example documents | مستندات تجريبية
│   ├── sample_arabic.pdf
│   └── sample_images/
└── outputs/                 # Generated files | الملفات المنتجة
```

---

## 🔧 Configuration | الإعدادات

### Required API Keys | مفاتيح API المطلوبة

**⚠️ Set these BEFORE running the system:**

1. **Hugging Face Token** (REQUIRED):
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx_your_huggingface_token_here"
```

2. **WandB API Key** (OPTIONAL):
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

**In Google Colab:**
```python
import os
# REPLACE WITH YOUR ACTUAL TOKENS:
HF_TOKEN = "hf_xxxxxxxxxxxxx_your_huggingface_token_here"
WANDB_API_KEY = "your_wandb_api_key_here"  # Optional

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
os.environ["WANDB_DISABLED"] = "true"  # Set to false if using WandB
```

**⚠️ Security Note**: Never share your tokens publicly!

### Key Parameters | المعاملات الرئيسية
- **max_tokens**: Maximum tokens for generation (default: 2000)
- **dpi**: PDF to image conversion resolution (default: 300)
- **font_size**: Output PDF font size (default: 14)

---

## 📊 Supported Formats | التنسيقات المدعومة

| Input | Output | Status |
|-------|--------|--------|
| PDF | TXT, PDF | ✅ Full Support |
| PNG | TXT, PDF | ✅ Full Support |
| JPG/JPEG | TXT, PDF | ✅ Full Support |
| TIF/TIFF | TXT, PDF | ✅ Full Support |
| BMP | TXT, PDF | ✅ Full Support |
| WEBP | TXT, PDF | ✅ Full Support |

---

## 🧪 Examples | أمثلة

### Example 1: Arabic Document Processing
```python
# Upload Arabic PDF document | رفع مستند عربي PDF
# Result: Extract text with proper Arabic shaping
# النتيجة: استخراج النص مع تشكيل عربي صحيح
```

### Example 2: Image Text Recognition
```python
# Process Arabic text in images | معالجة النص العربي في الصور
# Works with handwritten and printed text | يعمل مع النص المطبوع واليدوي
```

---

## 🐛 Troubleshooting | استكشاف الأخطاء

### Common Issues | المشاكل الشائعة

1. **API Authentication Errors** | أخطاء المصادقة لـ API
   ```
   Error: "401 Unauthorized" or "Invalid token"
   ```
   **Solution**:
   - Verify your Hugging Face token is correct
   - Ensure token has sufficient permissions
   - Check token hasn't expired

2. **Model Access Issues** | مشاكل الوصول للنموذج
   ```
   Error: "Model not found" or "Access denied"
   ```
   **Solution**:
   - Confirm you have access to `NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct`
   - Accept model terms on Hugging Face website
   - Check your Hugging Face account permissions

3. **Font Issues** | مشاكل الخط
   ```bash
   # Install Arabic fonts | تثبيت الخطوط العربية
   apt-get install fonts-dejavu-core fonts-liberation
   ```

4. **Memory Errors** | أخطاء الذاكرة
   ```python
   # Reduce batch size or use CPU | تقليل حجم الدفعة أو استخدام المعالج
   device = "cpu"  # Fallback to CPU
   ```

5. **WandB Connection Issues** | مشاكل اتصال WandB
   ```python
   # Disable WandB if having issues | تعطيل WandB إذا كانت هناك مشاكل
   os.environ["WANDB_DISABLED"] = "true"
   ```

6. **Network/Download Issues** | مشاكل الشبكة/التحميل
   ```
   Error: "Failed to download model"
   ```
   **Solution**:
   - Check internet connection
   - Verify Hugging Face is accessible
   - Try using a VPN if in restricted region

---

## 🤝 Contributing | المساهمة

We welcome contributions to improve Qari-OCR! | نرحب بالمساهمات لتحسين نظام قاري!

### How to Contribute | كيفية المساهمة:
1. Fork the repository | انشئ نسخة من المشروع
2. Create a feature branch | أنشئ فرع للمزايا الجديدة
3. Make your changes | اجرِ تغييراتك
4. Test thoroughly | اختبر بشكل شامل
5. Submit a pull request | أرسل طلب دمج

### Areas for Improvement | مجالات التحسين:
- 📱 Mobile app interface | واجهة تطبيق الهاتف
- 🗣️ Speech-to-text integration | التكامل مع تحويل الصوت لنص
- 🤖 Model fine-tuning | ضبط النموذج
- 🌐 Web interface | واجهة ويب

---

## 📄 License | الترخيص

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License
Copyright (c) 2025 Dr. Mohammed Tawfik
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 👨‍💻 Author | المؤلف

**Dr. Mohammed Tawfik**  
📧 Email: kmkhol01@gmail.com  
🔬 Research: Computer Vision & Arabic NLP  
🌍 Expertise: Arabic OCR, Document Analysis  

---

## 🙏 Acknowledgments | الشكر والتقدير

- **NAMAA-Space**: For the Qwen2VL Arabic model
- **Hugging Face**: For the Transformers library
- **Google Colab**: For the computational platform
- **Arabic NLP Community**: For tools and resources

---

## 📈 Roadmap | خارطة الطريق

### Version 1.1 (Planned)
- [ ] Web interface | واجهة ويب
- [ ] Batch processing optimization | تحسين المعالجة المجمعة
- [ ] Additional language support | دعم لغات إضافية

### Version 1.2 (Future)
- [ ] Mobile application | تطبيق الهاتف
- [ ] Cloud deployment | النشر السحابي
- [ ] Advanced text formatting | تنسيق نص متقدم

---

## 📞 Support | الدعم

For questions, bug reports, or feature requests:
- 📧 Email: kmkhol01@gmail.com
- 🐛 Issues: Create an issue on GitHub
- 💬 Discussions: Use GitHub Discussions

---

**Made with ❤️ for the Arabic NLP community**  
**صُنع بحب لمجتمع معالجة اللغة العربية**

---

*Last Updated: November 2025*  
*آخر تحديث: نوفمبر ٢٠٢٥*