package uncomplicate.neanderthal.protocols;

import java.nio.Buffer;

public interface Block {
    
    long length();

    long stride();
    
    Buffer buf ();
}
