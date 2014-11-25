package uncomplicate.neanderthal.protocols;

public interface Vector {
    
    long dim ();
    
    long iamax ();

    Vector swap (Vector y);

    Vector copy (Vector y);

    Vector segment (long k, long l);
}
