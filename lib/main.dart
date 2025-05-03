import 'package:flutter/material.dart';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Soul Chat',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue)),
      home: const SoulChatPage(title: 'Soul Chat'),
    );
  }
}

class SoulChatPage extends StatefulWidget {
  const SoulChatPage({super.key, required this.title});
  final String title;

  @override
  State<SoulChatPage> createState() => _SoulChatPageState();
}

class _SoulChatPageState extends State<SoulChatPage> {
  bool _isProcessing = false;
  bool _isDownloading = false;
  bool _isDownloadComplete = false;
  OrtSession? _session;
  String? _userInput = '';
  String _soulResponse = 'Hello, I am your soul. How can I assist you today?';
  List<String> _chatHistory = [];
  late String modelPath;
  String _errorMessage = '';

  final List<String> modelFiles = [
    "phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
    "phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
    "added_tokens.json",
    "config.json",
    "configuration_phi3.py",
    "genai_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
  ];

  final String baseUrl = "https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/";

  @override
  void initState() {
    super.initState();
    _setupModel();
  }

  Future<String> getModelDirectory() async {
    if (kIsWeb) {
      // For web-specific storage, use LocalStorage, IndexedDB, etc.
      return 'web-storage-path';  // This is a placeholder, adapt it for your use case
    } else {
      // For mobile platforms (iOS/Android), use path_provider
      Directory appDir = await getApplicationDocumentsDirectory();
      return appDir.path;
    }
  }

  Future<void> _setupModel() async {
    String directoryPath = await getModelDirectory();
    modelPath = "$directoryPath/phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx";

    // Start downloading the files
    setState(() {
      _isDownloading = true;
    });

    bool allFilesDownloaded = await _downloadModelFiles(directoryPath);

    if (allFilesDownloaded) {
      setState(() {
        _isDownloadComplete = true;
        _isDownloading = false;
      });
      // Load the model once the download is complete
      _session ??= await OnnxRuntime().createSession(modelPath);
    } else {
      setState(() {
        _isDownloading = false;
        _errorMessage = "Failed to download necessary files. Please try again.";
      });
    }
  }

  // Download model files and return true if all files are downloaded successfully
  Future<bool> _downloadModelFiles(String directoryPath) async {
    bool allFilesDownloaded = true;

    for (var file in modelFiles) {
      final filePath = "$directoryPath/$file";
      if (!File(filePath).existsSync()) {
        final url = "$baseUrl$file";
        try {
          final response = await http.get(Uri.parse(url));
          if (response.statusCode == 200) {
            await File(filePath).writeAsBytes(response.bodyBytes);
          } else {
            allFilesDownloaded = false;
            print("Failed to download: $file");
            break; // Stop downloading further files if one fails
          }
        } catch (e) {
          allFilesDownloaded = false;
          print("Error downloading $file: $e");
          break;
        }
      }
    }
    return allFilesDownloaded;
  }

  // Run inference for chat
  Future<void> _runChatInference() async {
    if (_userInput == null || _userInput!.isEmpty || _session == null) return;

    setState(() {
      _isProcessing = true;
    });

    _chatHistory.add('User: $_userInput');

    String context = _chatHistory.join('\n');
    String inputText = context + '\nSoul:';

    final inputData = Uint8List.fromList(inputText.codeUnits);
    final inputTensor = await OrtValue.fromList(inputData, [1, inputText.length]);

    final outputs = await _session!.run({'input': inputTensor});
    final responseTensor = outputs['output']!;
    final responseData = await responseTensor.asFlattenedList();

    String modelResponse = String.fromCharCodes(responseData.cast<int>());

    setState(() {
      _soulResponse = modelResponse;
      _chatHistory.add('Soul: $modelResponse');
      _isProcessing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(backgroundColor: Theme.of(context).colorScheme.inversePrimary, title: Text(widget.title)),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Display download progress or error message
            if (_isDownloading)
              const CircularProgressIndicator(),
            if (!_isDownloading && !_isDownloadComplete)
              Text(
                _errorMessage,
                style: const TextStyle(color: Colors.red),
              ),
            if (!_isDownloading && !_isDownloadComplete)
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _errorMessage = ''; // Clear any previous error
                  });
                  _setupModel();
                },
                child: const Text('Retry Download'),
              ),
            if (_isDownloadComplete) ...[
              // Display the chat history
              Expanded(
                child: ListView.builder(
                  itemCount: _chatHistory.length,
                  itemBuilder: (context, index) {
                    return ListTile(
                      title: Text(_chatHistory[index]),
                    );
                  },
                ),
              ),
              // Input text field for chatting with the soul
              TextField(
                onChanged: (value) => setState(() => _userInput = value),
                decoration: InputDecoration(
                  labelText: 'Type your message here...',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 10),
              // Send button to trigger inference
              ElevatedButton(
                onPressed: _isProcessing ? null : _runChatInference,
                child: _isProcessing
                    ? const CircularProgressIndicator()
                    : const Text('Send Message'),
              ),
              const SizedBox(height: 10),
              // Display the soul's response
              Text('Soul says: $_soulResponse', style: const TextStyle(fontWeight: FontWeight.bold)),
            ],
          ],
        ),
      ),
    );
  }
}
