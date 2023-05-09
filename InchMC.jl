using LinearAlgebra
using QuadGK
using Random
using Combinatorics

function Compute_bare(H0, si, sf, t, O)
    """
    Compute_bare computes the bare propagator on the unfolded Keldysh contour
    H0: Hamiltonian of the system
    s1, s2: Keldysh contour pair of time
    t: time of measurement
    O: operator to measure
    """
    if si<t && sf<t
        return exp(-1im*(sf-si)*H0)
    elseif t<=si && si<=sf
        return exp(1im*(sf-si)*H0)
    else
        return exp(1im*(sf-t)*H0)*O*exp(-1im*(t-si)*H0)
    end
end

function Compute_bold(ti, tf, si, sf, sup, t, H0, O, G)
    """
    Compute_alt computes the proper Green's functions
    si,sf,sup: times on the Keldysh contour
    ti, tf: Absolute boundaries of Keldysh diagrams
    t: measure time
    H0: Unperturbrd Hamiltonian
    O: Obserbable
    G: Collection of previously computed Green's functions
    """
    if si<=sf && sf<=sup
        return Compute_G(sf, si, tf, ti, G)
    elseif sup< si && si<=sf
        return Compute_bare(H0, si, sf, t, O)
    else       
        # In this case no interpolation is needed as the bold propagator is evaluated exactly at a point were it was
        # computed previously
        return Compute_bare(H0, sup, sf, t, O)*Compute_Gex(sup, si, G)
    end
end

function Compute_Gex(sup, si, G)
    iint = floor(Int, si/dt)+1
    fint = floor(Int, sup/dt)+1
    
    #println(sup)
    #println(si)
    #println(Access(G, iint, fint))
    #println(G)
    #println(iint)
    #println(fint)
    return Access(G,iint,fint)
end

function Compute_G(sf, si, tf, ti, G)
    """
    Compute_G performs the interpolation of bold Green's functions. Three cases must be distinguished
    sf, si: Keldysh time arguments of desired Green function interpolation
    tf, ti: Boundaries of the so far calculated Green's functions
    G: array of previously computed Green's functions
    """
    if abs(si-ti)<1e-8
        return Interpolate_Gi(si, sf, dt, G)
    elseif abs(sf-tf)<1e-8
        return Interpolate_Gf(si, sf, dt, G)
    else
        return Interpolate_G(si, sf, dt, G)
    end
end

function Interpolate_G(si, sf, dt, G)
    """
    Interpolate_G: compute the interpolation of bold Green function used for inchworm calculations
    si,sf: Keldysh contour time values
    ti, tf: Absolute Keldysh boundary time values
    dt: time interval
    G: array of pre-calculated Greens functions
    """
    # First thing to do is to find meaning full integers to interpolate from
    # HERE THE INDEXING HAS TO BE FIXED TO A 1D ARRAY
    iint = floor(Int,si/dt)+1
    fint = floor(Int,sf/dt)+1

    di = si-dt*(iint-1)
    df = sf-dt*(fint-1)

    return Access(G,iint,fint)+di*(Access(G,iint+1,fint)-Access(G,iint,fint))+df*(Access(G,iint,fint+1)-Access(G,iint,fint))+di*df*(Access(G,iint+1,fint+1)-Access(G,iint+1,fint)-Access(G,iint,fint+1)+Access(G,iint,fint))
end

function Access(G,iint, fint)
    mint1 = Int(0.5*(fint+1)*fint)
    mint2 = - iint+1
    return G[mint1+mint2]
end

function Interpolate_Gi(si, sf, dt, G)
    """
    Interpolate_Gi: computes the interpolation in the particular case where the initial point is the fisrt point in the interval
    si,sf: Keldysh time values
    dt: time interval
    G: array of Green's functions
    """
    fint= floor(Int,sf/dt)+1
    df = sf-dt*(fint-1)

    return Access(G,1,fint)+df*(Access(G,1,fint+1)-Access(G,1,fint))
end

function Interpolate_Gf(si, sf, dt, G)
    """
    Interpolate_Gi: computes the interpolation in the particular case where the initial point is the last point in the interval
    si,sf: Keldysh time values
    dt: time interval
    G: array of Green's functions
    """
    iint= floor(Int,si/dt)+1
    fint= floor(Int,sf/dt)+1
    di = sf-dt*(fint-1)

    return Access(G,iint,fint)+df*(Access(G,iint+1,fint)-Access(G,iint,fint))
end

function f(x,t1,t2,λ,ω,β)
    if x==0
        y =1e-7
        return λ/π*ω*y/(ω^2+y^2)*(coth(β*y/2)*cos(y*(t1-t2))-1im*sin(y*(t1-t2)))
    else
        return λ/π*ω*x/(ω^2+x^2)*(coth(β*x/2)*cos(x*(t1-t2))-1im*sin(x*(t1-t2)))
    end
end

function B(t1,t2,Bath)
    """
    B computes the bath correlation using semi-analytical formula
    t1, t2: times of correlations
    λ: reorganization energy
    ω: cutoff frequency
    β: inverse temperature
    Bath: array containing all previous parameters
    """
    λ, ω, β = Bath
    integral, error = quadgk(x -> f(x,t1,t2,λ,ω,β), -20*ω, 20*ω)
    return integral
end

function MC_diag(ti, dt, t, H0, W, O, Bath, M, Nwlk)
    """
    MC_diag: performs diagrammatic expansion using bare procedure
    ti: Initial time
    dt: time increment
    t: real time
    H0: unperturbed Hamiltonian
    W: interaction
    O: Observable
    Bath: Bath parameters
    Nwlk: Number of walkers
    M: order of diagrammatix expansion
    """
    tf = ti+dt
    # First approximation is to use free propagator
    G = Compute_bare(H0, ti, tf, t, O)
    # Now stochastic approches have to be used to compute higher order terms
    for i in 1:M
        # For a given order diagrams are stochastically summed
        G += Stochastic_sample(ti, tf, t, i, H0, W, O, Bath, Nwlk)
    end
    return G
end

function Stochastic_sample(ti, tf, t, M, H0, W, O, Bath, Nwlk)
    """
    Stochastic_sample: computes the integral of Dyson expansion using stochastic sample. Intervals are sampled using
    a regular grid.
    ti, tf: initial and final times of the propagation (these are times on the Keldysh contour)
    t: physical time
    M: order of integration (its really 2M as odd terms vanish)
    H0: free Hamiltonian
    W: perturbation
    O: Observable
    Bath: bath observable
    Nwlk: Number of sampling points
    """
    G_stoch = zeros(size(H0)[1],size(H0)[1])
    for i in 1:Nwlk
        # Here 2M random points on the interval ti-tf have to be sampled
        Xrand = ti.+(tf-ti).*rand(Float64, 2*M)
        Xrand = sort(Xrand)
        G_stoch += Compute_propagator(ti,tf,t,Xrand,H0,W,O,Bath)
    end
    # besides the average the integral value has to be corrected by the volume of the hyperspace of integration
    V = (tf-ti)^(2*M)/(fac(2*M))
    return V*G_stoch/Nwlk
end

function fac(i)
    if i ==1
        return 1
    else
        return i*fac(i-1)
    end
end

function Compute_propagator(ti,tf,t,Xrand,H0,W,O,Bath)
    """
    Compute_propagator calculates the propagator associated with a given stochastic sample of points Xrands
    ti, tf: initial and final points of Keldysh integration
    t: physical time
    Xrand: stochastic sample of points in the interval
    H0: unperturbed Hamiltonian
    W: perturbation
    O: obserbable
    Bath: class that includes bath parameters
    """
    # Non trivial step is to compute bath part as it scales horribly
    L = size(Xrand)[1]
    LBath = Compute_bath(Xrand, Bath)
    # First part of the propagations is done separately
    G = Compute_bare(H0, ti, Xrand[1], t, O)
    G = G * W
    for i in 2:L
        G = G*Compute_bare(H0, Xrand[i-1], Xrand[i], t, O)
        G = G * W
    end
    # Last part of the propagations is done separately
    G = G* Compute_bare(H0, last(Xrand), tf, t, O)
    # Now all boring prefactors have to be calculated
    prefact = 1im^L
    for i in 1:L
        if Xrand[i]<t
            prefact *=-1
        end
    end
    return prefact*G*LBath
end

function Compute_bath(Xrand, Bath)
    """
    Compute_bath calculates the bath contribution to the propagator
    Xrand: sampled collection of point in Keldysh contour
    Bath: bath parameters (are needed for semi analytical approach)
    """
    N = size(Xrand)[1]
    Partition = Pairwise_partitions(N)
    L = size(Partition)[1]
    Bath_influence = 0
    for i in 1:L
        Bath_influence += Compute_product(Xrand, Bath, Partition[i])
    end
    return Bath_influence
end

function Compute_product(Xrand, Bath, Partition)
    """
    Compute_product calculates the product of bath correlations for a given partition, semianalitical approach
    Xrand: sampled times on keldysh contour
    Bath: class containing bath parameters
    Partition: individual pair association of the sampled times
    """
    L = size(Partition)[1]
    G = 1
    for i in 1:L
        G *= B(Partition[i][1],Partition[i][2],Bath)
    end
    return G
end

function Pairwise_partitions(N)
    """
    Computes the pairwise partitions where every part has exactly two elements
    This is a brute force calculation and its very inefficient thus it should just be used to very low order expans-
    ion in N. Perhaps a (surprisingly) faster approach would be to use an iterative function nevertheless, I guess
    this is good enough for M=4 that in practive will be the maximum order of expansion once the full Inchworm algo-
    rithm is built
    N: length of the array to partiotionate
    """
    Npairs = floor(Int,N/2)
    A = partitions(1:N,Npairs)
    B = []
    for a in A
        flag =1
        for i in 1:Npairs
            if size(a[i])[1]!=2
                flag *=0
                break
            end
        end
        if flag !=0
            push!(B,a)
        end
    end
    return B
end

function InchDiag_MC(t, dt, H0, W, O, Bath, M, Nwlk)
    """
    InchDiag_MC implements Inchworm diagrammatic Monte Carlo using original approach
    t: physical time
    dt = time interval
    H0: Unpperturbed Hamiltonian
    W: perturbation
    O: overlap
    Bath: bath parameters
    M: order of Monte Carlo diagrammatic expansion
    Nwlk: Number of stochastic sampled points
    """
    # Inchworm requires to store a huge number of Green functions for previous times as resummation is needed in
    # order to avoid sign problem
    N= floor(Int, 2*t/dt)
    trange = range(0,2*t,N+1)
    # pre store array of Greens functions
    G = []
    Npar = 0
    for tf in trange
        Npar = Npar+1
        tirange = range(tf,0,Npar)
        for ti in tirange
            # Perform MC sampling using Inchworm algorithm
            println(tf)
            println(ti)
            G_par = Compute_Inchworm(ti, tf, dt, t, H0, W, O, G, Bath, M, Nwlk)
            push!(G, G_par)
        end
    end
    return last(G)
end

function Compute_Inchworm(ti, tf, dt, t, H0, W, O, G, Bath, M, Nwlk)
    """
    Compute_Inchworm calculates the Greens function for a given pair of Keldysh contour times. Inchworm algorithm
    is used to compute such Greens functions avoiding sign problem. As a consecuence, a sample of Green's functions
    have to be computed previously and interpolation techniques have to be deployed.
    ti, tf: Keldysh contour time values
    t: physical time
    dt: time interval
    H0: unperturbed Hamiltonian
    W: Perturbation
    O: observable
    G: Collection of previously computed Greens functions
    M: order of MC diagrammatic expansion
    Nwlk: Number of MC samples
    """
    if abs(tf-ti)<1e-8
        # This case is trivial as the Green function is simply the identity
        return I(2)
    elseif abs(tf-ti-dt)<1e-8
        # For the first non trivial step, plain MC samplig has to be implemented
        return MC_diag(ti, dt, t, H0, W, O, Bath, M, Nwlk)
    else
        # This is the only case were new functions have to be implemented as Inchworm algorithm is implemented in
        # this part
        return Inchworm_expand(ti, tf, dt, t, H0, W, O, G, Bath, M, Nwlk)
    end
end

function Inchworm_expand(ti, tf, dt, t, H0, W, O, G, Bath, M, Nwlk)
    """
    Inchworm_expand performs the stochastic sample of the Inchworm part of the algorithm
    ti, tf: Keldysh contour time values
    t: physical time
    H0: unperturbed Hamiltonian
    W: Perturbation
    O: observable
    G: Collection of previously computed Greens functions
    M: order of MC diagrammatic expansion
    Nwlk: Number of MC samples
    """
    # First approximation to the new propagator is gonna be the previous computed Greens function + free propagator
    # Note that given the if statements of the previous function last stored Greens function is always gonna be the
    # right one
    G_new = last(G) * Compute_bare(H0, tf-dt, tf, t, O)

    # Now diagrammatic expansion has to be considered
    for i in 1:M
        G_new = G_new + InchM_sample(ti, tf, dt, t, H0, W, O, G, Bath, i, Nwlk)
    end
    return G_new
end

function InchM_sample(ti, tf, dt, t, H0, W, O, G, Bath, M, Nwlk)
    """
    InchM_sample performs an stochastic calcualtion for a given order of diagrammatic expansion
    ti, tf: Keldysh contour time values
    t: physical time
    H0: unperturbed Hamiltonian
    W: Perturbation
    O: observable
    G: Collection of previously computed Greens functions
    M: order of MC diagrammatic expansion
    Nwlk: Number of MC samples
    """
    G_stoch = zeros(size(H0)[1],size(H0)[1])
    for i in 1:Nwlk
        # Here 2M random points on the interval ti-tf have to be sampled
        # For the Inchworm procedure, the sampling process is a bit more complex as the sampling must ensure
        # Inchworm properness so at least the first point has to be sampled within the interval [tf-dt,tf]
        Xrand = zeros(2*M)
        Xrand[1] = tf-dt+dt*rand(Float64)
        for i in 2:2*M
            Xrand[i] = ti+(tf-ti)*rand(Float64)
        end
        Xrand = sort(Xrand)
        G_stoch += Compute_Inchpropagator(ti, tf, dt, t, Xrand, H0, W, O, G, Bath)
    end
    # besides the average the integral value has to be corrected by the volume of the hyperspace of integration
    # CAREFUL perhaps this hypervolume has to be modified since the sampling method has changed
    V = (tf-ti)^(2*M)/(fac(2*M))
    return V*G_stoch/Nwlk
end

function Compute_Inchpropagator(ti, tf, dt, t, Xrand, H0, W, O, G, Bath)
    """
    Compute_Inchpropagator calculates the propagator asociated with a given sample of times on Keldysh contour
    ti,tf: limits of integration
    t: physical time
    Xrand: sample of times
    H0: Unperturbed Hamiltonian
    W: Perturbation
    O: Observable
    G: collection previously computed Greens functions
    """
    # The implementations should, in principle, be very similar to the one I previously computed.
    # Non trivial step is to compute bath part as it scales horribly
    L = size(Xrand)[1]
    # Only inchworm proper contractions have to be taken into account
    LBath = Compute_bathInch(Xrand, tf, dt, Bath)
    # First part of the propagations is done separately
    # Now, as here I'm resummating, the Green function is no longer made up of bare propagators but rather on bold
    # ones
    Gp = Compute_bold(ti, tf, ti, Xrand[1], tf-dt, t, H0, O, G)
    Gp = Gp * W
    for i in 2:L
        Gp = Gp*Compute_bold(ti, tf, Xrand[i-1], Xrand[i], tf-dt, t, H0, O, G)
        Gp = Gp * W
    end
    # Last part of the propagations is done separately
    Gp = Gp* Compute_bold(ti, tf, last(Xrand), tf, tf-dt, t, H0, O, G)
    # Now all boring prefactors have to be calculated
    prefact = 1im^L
    for i in 1:L
        if Xrand[i]<t
            prefact *=-1
        end
    end
    return prefact*Gp*LBath
end

function Compute_bathInch(Xrand, tf, dt, Bath)
    """
    Compute_bathInch computes the sum of possible factorizations that are Inchworm proper
    Xrand: sample of points
    tf,dt: time variables
    Bath: bath properties
    """
    N = size(Xrand)[1]
    Partition = Pairwise_partitions(N)
    L = size(Partition)[1]
    Bath_influence = 0
    for i in 1:L
        if Is_Inch(Partition[i], Xrand, dt, tf)
            Bath_influence += Compute_product(Xrand, Bath, Partition[i])
        end
    end
    return Bath_influence
end

function Is_Inch(Partition, Xrand, dt, tf)
    """
    Is_Inch verifies if a given partition is inchworm proper
    Partition: possible pairwise partition
    Xrand: sample of random Keldysh contour times
    dt, tf: bounday time arguments
    """
    ts = tf-dt
    L = size(Partition)[1]
    flag = 1
    for i in 1:L
        label_min = minimum(Partition[i])
        label_max = maximum(Partition[i])
        if Xrand[label_max]<ts
            # In this case inchworm properness have to be verifed
            if label_max==label_min+1
                flag *=0
                break
            end
        end
    end
    if flag==0
        return false
    else
        return true
    end
end

function main(dt, t, H0, W, O, Bath, M, Nwlk)
    """
    main functions just assembles everything
    dt: time step
    t: total time propagation
    H0: Unperturbed Hamiltonian
    W: Perturbation
    O: observable
    Bath: bath parameters
    M: order of diagrammatic expansion
    Nwlk: Number or walkers
    """
    Nsims = floor(Int,t/dt)
    G = []
    for i in 1:Nsims
        push!(G, InchDiag_MC(i*dt,dt,H0,W,O,Bath,M,Nwlk))
    end
    return G
end


t=1
dt = 0.1
H0 = zeros(2,2)
H0[1,1] = 0.5
H0[1,2] = 0.1
H0[2,1] = 0.1
H0[2,2] = -0.5
W = zeros(2,2)
W[1,1] = 0.5
W[2,2] = -0.5
O = W
M=1
Nwlk = 100
Bath=[1,1,1]
G=main(dt, t, H0, W, O, Bath, M, Nwlk)

