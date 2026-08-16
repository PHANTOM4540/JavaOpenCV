import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;

public class SmoothFaceTracker extends JPanel {
    private BufferedImage currentFrame;
    private byte[] reusablePixelBuffer;

    // Temporal Persistence & Smoothing Variables
    private static Rect lastTrackedBox = null;
    private static int framesMissing = 0;
    private static final int MAX_PERSIST_FRAMES = 10; // Keep box alive for 10 missed frames
    private static final double SMOOTHING_FACTOR = 0.4; // Position smoothing (0.0 = static, 1.0 = instant)

    private static final String YUNET_FILE = "face_detection_yunet_2023mar.onnx";
    private static final String[] YUNET_MIRRORS = {
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/yunet/resolve/main/face_detection_yunet_2023mar.onnx"
    };

    private static final String FRONTAL_XML = "haarcascade_frontalface_alt.xml";
    private static final String PROFILE_XML = "haarcascade_profileface.xml";
    private static final String FRONTAL_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_alt.xml";
    private static final String PROFILE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_profileface.xml";

    public static void main(String[] args) {
        System.out.println("=== Starting Flicker-Free Smooth Face Tracker ===");

        OpenCV.loadLocally();

        FaceDetectorYN dnnDetector = prepareYuNetDetector();
        CascadeClassifier frontalCascade = null;
        CascadeClassifier profileCascade = null;

        if (dnnDetector == null) {
            downloadFileIfMissing(FRONTAL_XML, FRONTAL_URL, 50000);
            downloadFileIfMissing(PROFILE_XML, PROFILE_URL, 30000);
            frontalCascade = new CascadeClassifier(FRONTAL_XML);
            profileCascade = new CascadeClassifier(PROFILE_XML);
        }

        VideoCapture camera = new VideoCapture();
        camera.open(0, Videoio.CAP_DSHOW);
        if (!camera.isOpened()) camera.open(0, Videoio.CAP_ANY);

        if (!camera.isOpened()) {
            System.err.println("[ERROR] Unable to access camera.");
            return;
        }

        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 640);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 480);
        camera.set(Videoio.CAP_PROP_FPS, 60);

        Mat frame = new Mat();
        camera.read(frame);
        if (frame.empty()) return;

        Mat smallFrame = new Mat();
        double scale = 0.5;
        Size detectSize = new Size(frame.cols() * scale, frame.rows() * scale);

        if (dnnDetector != null) {
            dnnDetector.setInputSize(detectSize);
        }

        JFrame window = new JFrame("Smooth Flicker-Free Face Tracker");
        SmoothFaceTracker panel = new SmoothFaceTracker();
        window.setContentPane(panel);
        window.setSize(frame.cols(), frame.rows());
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setLocationRelativeTo(null);
        window.setVisible(true);

        Mat faces = new Mat();
        Mat graySmall = new Mat();
        Mat grayFlipped = new Mat();
        MatOfRect frontalFaces = new MatOfRect();
        MatOfRect profileRight = new MatOfRect();
        MatOfRect profileLeft = new MatOfRect();

        while (camera.read(frame)) {
            if (frame.empty()) continue;

            Imgproc.resize(frame, smallFrame, detectSize);
            Rect currentRawDetection = null;

            if (dnnDetector != null) {
                // Run 360° AI (Lower confidence threshold to 0.45f to prevent dropouts)
                dnnDetector.detect(smallFrame, faces);

                if (faces.rows() > 0) {
                    double invScale = 1.0 / scale;
                    int x = (int) (faces.get(0, 0)[0] * invScale);
                    int y = (int) (faces.get(0, 1)[0] * invScale);
                    int w = (int) (faces.get(0, 2)[0] * invScale);
                    int h = (int) (faces.get(0, 3)[0] * invScale);
                    currentRawDetection = new Rect(x, y, w, h);
                }
            } else {
                // Fallback Cascade Engine
                Imgproc.cvtColor(smallFrame, graySmall, Imgproc.COLOR_BGR2GRAY);
                Imgproc.equalizeHist(graySmall, graySmall);
                double invScale = 1.0 / scale;

                if (frontalCascade != null && !frontalCascade.empty()) {
                    frontalCascade.detectMultiScale(graySmall, frontalFaces, 1.1, 3, Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
                    if (frontalFaces.toArray().length > 0) {
                        Rect r = frontalFaces.toArray()[0];
                        currentRawDetection = new Rect((int)(r.x * invScale), (int)(r.y * invScale), (int)(r.width * invScale), (int)(r.height * invScale));
                    }
                }

                if (currentRawDetection == null && profileCascade != null && !profileCascade.empty()) {
                    profileCascade.detectMultiScale(graySmall, profileRight, 1.1, 3, Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
                    if (profileRight.toArray().length > 0) {
                        Rect r = profileRight.toArray()[0];
                        currentRawDetection = new Rect((int)(r.x * invScale), (int)(r.y * invScale), (int)(r.width * invScale), (int)(r.height * invScale));
                    } else {
                        Core.flip(graySmall, grayFlipped, 1);
                        profileCascade.detectMultiScale(grayFlipped, profileLeft, 1.1, 3, Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
                        if (profileLeft.toArray().length > 0) {
                            Rect r = profileLeft.toArray()[0];
                            int mappedX = (int) ((smallFrame.cols() - r.x - r.width) * invScale);
                            currentRawDetection = new Rect(mappedX, (int)(r.y * invScale), (int)(r.width * invScale), (int)(r.height * invScale));
                        }
                    }
                }
            }

            // --- SMOOTHING & PERSISTENCE LOGIC ---
            if (currentRawDetection != null) {
                framesMissing = 0;
                if (lastTrackedBox == null) {
                    lastTrackedBox = currentRawDetection;
                } else {
                    // Exponential Moving Average (EMA) position smoothing
                    int smoothX = (int) (lastTrackedBox.x + SMOOTHING_FACTOR * (currentRawDetection.x - lastTrackedBox.x));
                    int smoothY = (int) (lastTrackedBox.y + SMOOTHING_FACTOR * (currentRawDetection.y - lastTrackedBox.y));
                    int smoothW = (int) (lastTrackedBox.width + SMOOTHING_FACTOR * (currentRawDetection.width - lastTrackedBox.width));
                    int smoothH = (int) (lastTrackedBox.height + SMOOTHING_FACTOR * (currentRawDetection.height - lastTrackedBox.height));
                    lastTrackedBox = new Rect(smoothX, smoothY, smoothW, smoothH);
                }
            } else {
                // If detection missed a frame, keep last box active for MAX_PERSIST_FRAMES
                framesMissing++;
                if (framesMissing > MAX_PERSIST_FRAMES) {
                    lastTrackedBox = null; // Reset only if face is gone for > 10 frames (~0.3s)
                }
            }

            // Draw smoothed green bounding box
            if (lastTrackedBox != null) {
                Imgproc.rectangle(frame, lastTrackedBox, new Scalar(0, 255, 0), 3);
                Imgproc.putText(frame, "Tracking Active", new Point(lastTrackedBox.x, Math.max(lastTrackedBox.y - 10, 20)),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 0), 2);
            }

            panel.updateImageOptimized(frame);
        }

        camera.release();
    }

    private void updateImageOptimized(Mat mat) {
        int width = mat.cols(), height = mat.rows(), channels = mat.channels();
        int requiredSize = width * height * channels;

        if (reusablePixelBuffer == null || reusablePixelBuffer.length != requiredSize) {
            reusablePixelBuffer = new byte[requiredSize];
            currentFrame = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        }

        mat.get(0, 0, reusablePixelBuffer);
        byte[] targetData = ((DataBufferByte) currentFrame.getRaster().getDataBuffer()).getData();
        System.arraycopy(reusablePixelBuffer, 0, targetData, 0, requiredSize);

        repaint();
    }

    private static FaceDetectorYN prepareYuNetDetector() {
        File file = new File(YUNET_FILE);
        if (file.exists() && file.length() < 200000) file.delete();

        if (!file.exists()) {
            HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.ALWAYS).build();
            for (String urlStr : YUNET_MIRRORS) {
                try {
                    HttpRequest req = HttpRequest.newBuilder().uri(URI.create(urlStr)).header("User-Agent", "Mozilla/5.0").GET().build();
                    HttpResponse<InputStream> resp = client.send(req, HttpResponse.BodyHandlers.ofInputStream());
                    if (resp.statusCode() == 200) {
                        Files.copy(resp.body(), file.toPath());
                        if (file.length() > 200000) break;
                        else file.delete();
                    }
                } catch (Exception ignored) {}
            }
        }

        if (file.exists() && file.length() > 200000) {
            try { 
                // Lower score threshold to 0.40f for maximum sensitivity
                return FaceDetectorYN.create(file.getAbsolutePath(), "", new Size(320, 320), 0.40f, 0.3f, 5000); 
            } catch (Exception ignored) {}
        }
        return null;
    }

    private static void downloadFileIfMissing(String fileName, String urlStr, long minExpectedSize) {
        File file = new File(fileName);
        if (!file.exists() || file.length() < minExpectedSize) {
            try {
                HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.ALWAYS).build();
                HttpRequest req = HttpRequest.newBuilder().uri(URI.create(urlStr)).header("User-Agent", "Mozilla/5.0").GET().build();
                HttpResponse<InputStream> resp = client.send(req, HttpResponse.BodyHandlers.ofInputStream());
                if (resp.statusCode() == 200) Files.copy(resp.body(), file.toPath());
            } catch (Exception ignored) {}
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (currentFrame != null) {
            g.drawImage(currentFrame, 0, 0, getWidth(), getHeight(), null);
        }
    }
}