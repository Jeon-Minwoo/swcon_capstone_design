package solutionanalysis.util.networking;

import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ConnectionManager {

    private static final String TAG = "ConnectionManager";

    private static final String HOST_ADDRESS = "192.168.0.2";
    private static final int PORT = 58431;

    private static final int BUFFER_SIZE = 1024;
    private static final byte CLIENT_REQUEST_ID = (byte) 255;

    private static final String SENDER_THREAD_NAME = "solubilityanalysis.util.networking.ConnectionManager_sender";
    private static final String RECEIVER_THREAD_NAME = "solubilityanalysis.util.networking.ConnectionManager_receiver";

    public static ConnectionManager getInstance() {

        if (instance == null) {

            instance = new ConnectionManager();
        }
        return instance;
    }

    public static ConnectionManager instance = null;

    private Socket socket = null;
    private HandlerThread senderThread = null;
    private Thread receiverThread = null;

    private IInteractionHandler interactionHandler = null;
    private IConnectionErrorListener connectionErrorListener = null;

    private RequestBundle lastRequestedBundle = null;

    private ConnectionManager() {

        initThread();
    }

    private void initThread() {

        senderThread = new HandlerThread(SENDER_THREAD_NAME);
        receiverThread = new Thread(RECEIVER_THREAD_NAME) {
            @Override
            public void run() {

                //noinspection InfiniteLoopStatement
                while (true) {
                    if (socket == null) {
                        try {
                            //noinspection BusyWait
                            Thread.sleep(500);
                            Log.v(TAG, "ReceiverThread: No Socket, sleep()");
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    } else {
                        Log.v(TAG, "ReceiverThread: Ready to Receive");
                        try {
                            InputStream iStream = socket.getInputStream();
                            Log.v(TAG, "Stream opened");

                            byte[] miniBuffer = new byte[BUFFER_SIZE];
                            //noinspection ResultOfMethodCallIgnored
                            iStream.read(miniBuffer);
                            int dataLength = ByteBuffer.wrap(miniBuffer).getInt();

                            ByteArrayOutputStream variableBuffer = new ByteArrayOutputStream();
                            while (variableBuffer.size() < dataLength) {

                                int length = iStream.read(miniBuffer);
                                if (length > 0) {
                                    variableBuffer.write(miniBuffer, 0, length);
                                }
                            }
                            byte[] buffer = variableBuffer.toByteArray();

                            if (interactionHandler != null) {
                                Log.d(TAG, "InteractionHandler != null");
                                RequestBundle bundle = RequestBundle.fromBytes(buffer);
                                Log.d(TAG, "ReceiverThread: " + (bundle.response == null));
                                if (bundle.requestId == CLIENT_REQUEST_ID) {
                                    // Got a response
                                    interactionHandler.handleResponse(bundle);
                                } else {
                                    // Got a request
                                    interactionHandler.handleRequest(ConnectionManager.this, bundle);
                                }
                            }
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        };
    }

    public IInteractionHandler getInteractionHandler() {

        return interactionHandler;
    }

    public void setInteractionHandler(IInteractionHandler handler) {

        this.interactionHandler = handler;
    }

    public IConnectionErrorListener getConnectionErrorListener() {

        return connectionErrorListener;
    }

    public void setConnectionErrorListener(IConnectionErrorListener listener) {

        this.connectionErrorListener = listener;
    }

    public void connect(ERequest role) {

        Log.d(TAG, "connect()");

        senderThread.start();
        receiverThread.start();

        if (isConnected()) {
            Log.d(TAG, "connect(), Already Connected");
            connectionErrorListener.onAlreadyConnected();
        } else {
            new Handler(senderThread.getLooper()).post(() -> {
                Log.d(TAG, "connect(), post()");
                try {
                    // Connect as role
                    Log.d(TAG, "connect(), Connecting...");
                    socket = new Socket(HOST_ADDRESS, PORT);
                    socket.setSendBufferSize(BUFFER_SIZE);
                    socket.setReceiveBufferSize(BUFFER_SIZE);

                    Log.d(TAG, "connect(), Request");
                    request(role, new byte[0]);
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.d(TAG, "connect(), Host Unreachable");
                }
            });
        }
    }

    public boolean isConnected() {

        if (socket == null) {
            return false;
        } else {
            return socket.isConnected();
        }
    }

    public boolean disconnect() {

        if (isConnected()) {

            boolean result = request(ERequest.ANY_QUIT, new byte[0]) != null;

            receiverThread.interrupt();
            senderThread.interrupt();
            initThread();

            socket = null;
            return result;
        } else {

            socket = null;
            return false;
        }
    }

    public RequestBundle request(ERequest request, byte[] args) {

        if (lastRequestedBundle == null) {

            RequestBundle bundle = new RequestBundle(CLIENT_REQUEST_ID, request, args, EResponse.NONE);
            writeBundle(bundle);

            return bundle;
        }
        return null;
    }

    public void response(RequestBundle bundle) {

        writeBundle(bundle);
    }

    private void writeBundle(RequestBundle bundle) {

        byte[] data = bundle.toBytes();
        ByteBuffer sizeBuffer = ByteBuffer.allocate(Integer.BYTES);
        sizeBuffer.order(ByteOrder.BIG_ENDIAN);
        sizeBuffer.putInt(data.length);

        new Handler(senderThread.getLooper()).post(() -> {
            try {
                OutputStream oStream  = socket.getOutputStream();
                oStream.write(sizeBuffer.array());
                oStream.write(data);
                oStream.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }
}
