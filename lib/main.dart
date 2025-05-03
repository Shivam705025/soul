import 'package:flutter/material.dart';
import 'dart:async';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter/services.dart';

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
  OrtSession? _session;
  final assetPath = 'assets/models/phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx';  // Change to your model path
  String? _userInput = '';
  String _soulResponse = 'Hello, I am your soul. How can I assist you today?';
  List<String> _chatHistory = []; // To keep track of the conversation

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  // Initialize the ONNX model
  Future<void> _loadModel() async {
    _session ??= await OnnxRuntime().createSessionFromAsset(assetPath);
  }

  // Method to run inference and simulate chat
  Future<void> _runChatInference() async {
    if (_userInput == null || _userInput!.isEmpty) return;

    setState(() {
      _isProcessing = true;
    });

    // Append the user input to chat history to maintain context
    _chatHistory.add('User: $_userInput');

    // Prepare the context for the chat, combining history
    String context = _chatHistory.join('\n');
    String inputText = context + '\nSoul:';

    // Prepare input for the model
    final inputData = Uint8List.fromList(inputText.codeUnits);
    final inputTensor = await OrtValue.fromList(inputData, [1, inputText.length]);

    // Run inference
    final outputs = await _session!.run({'input': inputTensor});

    // Get the output from the model (response)
    final responseTensor = outputs['output']!;
    final responseData = await responseTensor.asFlattenedList();

    // Convert response data to string (model output)
    String modelResponse = String.fromCharCodes(responseData.cast<int>());

    // Update chat history with the soul's response
    setState(() {
      _soulResponse = modelResponse;  // The soul's response
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
              onChanged: (value) {
                setState(() {
                  _userInput = value;
                });
              },
              decoration: InputDecoration(
                labelText: 'Type your message here...',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 10),
            // Send button to trigger inference
            ElevatedButton(
              onPressed: _isProcessing
                  ? null
                  : () {
                      _runChatInference();
                    },
              child: _isProcessing
                  ? const CircularProgressIndicator()
                  : const Text('Send Message'),
            ),
            const SizedBox(height: 10),
            // Display the soul's response
            Text('Soul says: $_soulResponse', style: const TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }
}
