import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import "package:image/image.dart" as img;

class Detection {
  final double x;
  final double y;
  final double w;
  final double h;
  final double confidence;

  Detection(this.x, this.y, this.w, this.h, this.confidence);
}


late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  cameras = await availableCameras();

  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late Interpreter interpreter;
  late CameraController controller;
  List<Detection> detections = [];
  int chickenCount = 0;
  bool isProcessing = false;
  bool isCameraInitialized = false;

  @override
  void initState() {
    super.initState();
    initializeApp();
  }

  Future<void> initializeApp() async {
    await loadModel();

    controller = CameraController(
      cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await controller.initialize();

    controller.startImageStream((CameraImage image) {
      runModel(image);
    });

    if (mounted) {
      setState(() {
        isCameraInitialized = true;
      });
    }
  }

  img.Image convertCameraImage(CameraImage cameraImage) {
  final width = cameraImage.width;
  final height = cameraImage.height;

  final imgImage = img.Image(width: width, height: height);

  final plane = cameraImage.planes[0];

  int pixelIndex = 0;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final pixel = plane.bytes[pixelIndex];

      imgImage.setPixelRgb(x, y, pixel, pixel, pixel);

      pixelIndex++;
    }
  }

  return imgImage;
}


List<List<List<List<double>>>> imageToTensor(img.Image image, int size) {

  img.Image resized = img.copyResize(
    image,
    width: size,
    height: size,
  );

  var input = List.generate(
    1,
    (_) => List.generate(
      size,
      (y) => List.generate(
        size,
        (x) {
          var pixel = resized.getPixel(x, y);

         return [
            pixel.r / 255.0,
            pixel.g / 255.0,
            pixel.b / 255.0,
          ];
        },
      ),
    ),
  );

  return input;
}


  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset(
      'assets/models/best_float16.tflite',
    );

    print("YOLO Model Loaded");

    var inputShape = interpreter.getInputTensor(0).shape;
    var outputShape = interpreter.getOutputTensor(0).shape;

    print("Input Shape: $inputShape");
    print("Output Shape: $outputShape");
  }

  void runModel(CameraImage image) async {

  if (isProcessing) return;
  isProcessing = true;

  int inputSize = interpreter.getInputTensor(0).shape[1];

  img.Image convertedImage = convertCameraImage(image);

  var input = imageToTensor(convertedImage, inputSize);

  var outputShape = interpreter.getOutputTensor(0).shape;

  var output = List.generate(
    outputShape[0],
    (_) => List.generate(
      outputShape[1],
      (_) => List.filled(outputShape[2], 0.0),
    ),
  );

  interpreter.run(input, output);

  List<Detection> newDetections = [];

  for (int i = 0; i < outputShape[2]; i++) {

    double confidence = output[0][4][i];

    if (confidence > 0.5) {

      double x = output[0][0][i];
      double y = output[0][1][i];
      double w = output[0][2][i];
      double h = output[0][3][i];

      newDetections.add(
        Detection(x, y, w, h, confidence),
      );
    }
  }

  setState(() {
    detections = newDetections;
    chickenCount = newDetections.length;
  });

  isProcessing = false;
}
  @override
  void dispose() {
    controller.dispose();
    interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!isCameraInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Chicken Counter Camera"),
      ),
      body: Stack(
  children: [

    CameraPreview(controller),

    CustomPaint(
      painter: DetectionPainter(detections),
      child: Container(),
    ),

    Positioned(
      top: 50,
      left: 20,
      child: Container(
        padding: const EdgeInsets.all(10),
        color: Colors.black54,
        child: Text(
          "Chickens: $chickenCount",
          style: const TextStyle(
            color: Colors.white,
            fontSize: 24,
          ),
        ),
      ),
    ),

  ],
),
    );
  }
}


class DetectionPainter extends CustomPainter {

  final List<Detection> detections;

  DetectionPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {

    final paint = Paint()
      ..color = Colors.green
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    for (var det in detections) {

      double left = det.x * size.width;
      double top = det.y * size.height;
      double width = det.w * size.width;
      double height = det.h * size.height;

      canvas.drawRect(
        Rect.fromLTWH(left, top, width, height),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}