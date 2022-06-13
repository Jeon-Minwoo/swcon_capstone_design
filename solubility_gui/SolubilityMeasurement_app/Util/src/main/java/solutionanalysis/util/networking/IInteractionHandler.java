package solutionanalysis.util.networking;

import java.io.OutputStream;

public interface IInteractionHandler {

    void handleRequest(ConnectionManager conn, RequestBundle bundle);

    void handleResponse(RequestBundle bundle);
}
