package solutionanalysis.util.networking;

public interface IConnectionErrorListener {

    void onAlreadyConnected();

    void onHostUnreachable();
}
