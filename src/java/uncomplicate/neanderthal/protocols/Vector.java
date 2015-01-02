package uncomplicate.neanderthal.protocols;

public interface Vector {

    long dim ();

    long iamax ();

    Vector subvector  (long k, long l);

    Vector rotg ();

    Vector rotmg (Vector args);

    Vector rotm (Vector y, Vector p);

}
