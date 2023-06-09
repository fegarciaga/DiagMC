using LinearAlgebra
using QuadGK
using Random
using Combinatorics
using DelimitedFiles

function Compute_bare(H0, si, sf, t, O)
    """
    Compute_bare computes the bare propagator on the unfolded Keldysh contour
    H0: Hamiltonian of the system
    s1, s2: Keldysh contour pair of time
    t: time of measurement
    O: operator to measure
    """
    if si<=sf && sf<t
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
        return Compute_G(sf, si, tf, ti, t, O, G)
    elseif sup< si && si<=sf
        return Compute_bare(H0, si, sf, t, O)
    else       
        # In this case no interpolation is needed as the bold propagator is evaluated exactly at a point were it was
        # computed previously
        return Compute_bare(H0, sup, sf, t, O)*Interpolate_Gf(si, sup, dt, t, O, G)
    end
end

function Compute_Gex(sup, si, G)
    iint = floor(Int, si/dt)+1
    fint = floor(Int, sup/dt)+1
    
    return Access(G,iint,fint)
end

function Compute_G(sf, si, tf, ti, t, O, G)
    """
    Compute_G performs the interpolation of bold Green's functions. Three cases must be distinguished
    sf, si: Keldysh time arguments of desired Green function interpolation
    tf, ti: Boundaries of the so far calculated Green's functions
    G: array of previously computed Green's functions
    """
    if abs(si-ti)<1e-8
        return Interpolate_Gi(si, sf, dt, t, O, G)
    elseif abs(sf-tf)<1e-8
        return Interpolate_Gf(si, sf, dt, t, O, G)
    else
        return Interpolate_G(si, sf, dt, t, O, G)
    end
end

function Interpolate_G(si, sf, dt, t, O, G)
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

    return Access(G,iint,fint)+di/dt*(Access(G,iint+1,fint)-Access(G,iint,fint))+df/dt*(Access(G,iint,fint+1)-Access(G,iint,fint))+di*df*(Access(G,iint+1,fint+1)-Access(G,iint+1,fint)-Access(G,iint,fint+1)+Access(G,iint,fint))
end

function Access(G,iint, fint)
    mint1 = Int(0.5*(fint+1)*fint)
    mint2 = - iint+1
    return G[mint1+mint2]
end

function Interpolate_Gi(si, sf, dt, t, O, G)
    """
    Interpolate_Gi: computes the interpolation in the particular case where the initial point is the fisrt point in the interval
    si,sf: Keldysh time values
    dt: time interval
    G: array of Green's functions
    """
    fint= floor(Int,sf/dt)+1
    df = sf-dt*(fint-1)

    # If the interpolation has to use the point were the contour folds attention has to be put

    tint = floor(Int, t/dt)+1

    if tint == fint+1
        Gp = Access(G,1,fint+1)*inv(O)
        G_inter = Access(G,1,fint)+df/dt*(Gp-Access(G,1,fint))
    else
        G_inter = Access(G, 1, fint)+df/dt*(Access(G,1,fint+1)-Access(G,1,fint))
    end

    return G_inter
end

function Interpolate_Gf(si, sf, dt, t, O, G)
    """
    Interpolate_Gi: computes the interpolation in the particular case where the initial point is the last point in the interval
    si,sf: Keldysh time values
    dt: time interval
    G: array of Green's functions
    """
    iint= floor(Int,si/dt)+1
    fint= floor(Int,sf/dt)+1 
    di = si-dt*(iint-1)
    
    # Again, same careful analysis has to be used 
    
    tint = floor(Int, t/dt)+1

    if tint == iint
        Gp = Access(G,iint,fint)
        G_inter = Gp+di/dt*(Access(G,iint+1,fint)-Gp)
    else
        G_inter = Access(G,iint,fint)+di/dt*(Access(G,iint+1,fint)-Access(G,iint,fint))
    end

    return G_inter
end

function f(x,t1,t2,λ,ω,β)
    if x==0
        y =1e-14
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
    integral, error = quadgk(x -> f(x,t1,t2,λ,ω,β), 0, 50*ω)
    return integral
end

function MC_diag(ti, dt, t, H0, W, O, Bath, M, psi0, Nwlk)
    """
    MC_diag: performs diagrammatic expansion using bare procedure
    ti: Initial time
    dt: time increment
    t: real time
    H0: unperturbed Hamiltonian
    W: interaction
    O: Observable
    Bath: Bath parameters
    psi0: initial density matrix
    Nwlk: Number of walkers
    M: order of diagrammatix expansion
    """
    tf = ti+dt
    # First approximation is to use free propagator
    G = Compute_bare(H0, ti, tf, t, O)
    # Now stochastic approches have to be used to compute higher order terms
    for i in 1:M
        # For a given order diagrams are stochastically summed
        G += Stochastic_sample(ti, tf, t, i, H0, W, O, Bath, psi0, Nwlk)
    end
    return G
end

function Stochastic_sample(ti, tf, t, M, H0, W, O, Bath, psi0, Nwlk)
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
    psi0: initial density matrix
    Nwlk: Number of sampling points
    """
    G_stoch = zeros(size(H0)[1],size(H0)[1])
    # Initially just one random statring multidimensional array has to be sampled
    # Here 2M random points on the interval tf-ti have to sampled
    Xrand = ti.+(tf-ti).*rand(Float64, 2*M)
    Xrand = sort(Xrand)
    # In order to increase performance the initial probability is precomputed
    Gpar =  Compute_propagator(ti,tf,t,Xrand,H0,W,O,Bath)
    Pi = abs(tr(Gpar*psi0))
    Norm_fact = 0
    for i in 1:Nwlk
        # Here 2M random points on the interval ti-tf have to be sample
        # In order to improve the convergence of the integral Metropolis Hastings algorithm has to be implemented
        # This will vary the array Xrand until it finds an optimal representation of the probability distribution 
        # which is given by |O|
        Xrandalt, P = Metropolis_Hasting(ti, tf, t, H0, W, O, Bath, Xrand, Pi, psi0)
        G_stoch += Compute_propagator(ti,tf,t,Xrandalt,H0,W,O,Bath)/P
        Norm_fact += 1/P
    end
    # besides the average the integral value has to be corrected by the volume of the hyperspace of integration
    # This volume is trickier than it seems as several possible partitions (and thus integrals) are actually being computed
    if ti<t && t<tf
        # If the integral includes two parts of the contour several possible partitions have to be considered
        V=0
        for i in 0:2*M
            V+= (t-ti)^i/fac(i)*(tf-t)^(2*M-i)/fac(2*M-i)
        end
    else
        # If not the volume is simply calculated using a single hypervolume formula
        V = (tf-ti)^(2*M)/(fac(2*M))
    end
    Norm_fact/=Nwlk
    return V*G_stoch/(Nwlk*Norm_fact)
end

function Metropolis_Hasting(ti, tf, t, H0, W, O, Bath, Xrand, Pi, psi0)
    """
    Metropolis_Hasting improve the stochastic representation by means of the Metropolis Hastings algortihm
    ti,tf,t: time parameters
    H0: Unperturbed Hamiltonian
    W: Perturbation
    O: Observable
    Bath: Bath parameters
    Xrand: Initial point for the random walk
    Pi: associated probability density with the initial random point
    """
    Pold= Pi
    Pnew = 0
    # First thing to do is to propose a random walk
    L = size(Xrand)[1]
    Xrandold=Xrand
    Xrandnew  = zeros(L)
    for i in 1:50
        drand = 10*dt.*(0.5.-rand(Float64,L))
        Xrand1 = ti.+(mod.((Xrandold + drand).-ti, tf-ti))
        Xrand1 = sort(Xrand1)
        # once the new proposed state is has been computed probabilistic change is computed
        coin_flip = rand(Float64)
        Gnew = Compute_propagator(ti,tf,t,Xrand1,H0,W,O,Bath)
        Pnew = abs(tr(Gnew*psi0))
        if Pnew/Pold>1
            Xrandold = Xrand1
            Pold = Pnew
        elseif coin_flip<=Pnew/Pold
            Xrandnold = Xrand1
            Pold = Pnew
        end
    end
    # Return both the probability and the new sampling as the first one is used in Metropolis Hasting normalization
    return Xrandold, Pold
end
    
function fac(i)
    if i ==0
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
    LBath = Compute_bath(Xrand, Bath, t)
    # First part of the propagations is done separately
    G = Compute_bare(H0, ti, Xrand[1], t, O)
    G = W * G
    for i in 2:L
        G = Compute_bare(H0, Xrand[i-1], Xrand[i], t, O) * G
        G = W * G
    end
    # Last part of the propagations is done separately
    G = Compute_bare(H0, last(Xrand), tf, t, O) * G
    # Now all boring prefactors have to be calculated
    prefact = 1im^L
    for i in 1:L
        if Xrand[i]<t
            prefact *=-1
        end
    end
    return prefact*G*LBath
end

function Compute_bath(Xrand, Bath, t)
    """
    Compute_bath calculates the bath contribution to the propagator
    Xrand: sampled collection of point in Keldysh contour
    Bath: bath parameters (are needed for semi analytical approach)
    t: physical time
    """
    N = size(Xrand)[1]
    Partition = Pairwise_partitions(N)
    L = size(Partition)[1]
    Bath_influence = 0
    for i in 1:L
        Bath_influence += Compute_product(Xrand, Bath, Partition[i], t)
    end
    return Bath_influence
end

function Compute_product(Xrand, Bath, Partition, t)
    """
    Compute_product calculates the product of bath correlations for a given partition, semianalitical approach
    Xrand: sampled times on keldysh contour
    Bath: class containing bath parameters
    Partition: individual pair association of the sampled times
    t: physical time
    """
    L = size(Partition)[1]
    G = 1
    for i in 1:L
        # time has to be properly fixed
        t1 = Compute_time(Xrand[Partition[i][1]],t)
        t2 = Compute_time(Xrand[Partition[i][2]],t)
        G *= B(t1,t2,Bath)
    end
    return G
end

function Compute_time(s,t)
    """
    Compute_time: transform to meaningful time for bath correlation calcualtions
    s: time on th eun folded contour
    t: physical time
    """
    if s<t
        return s
    else
        return 2*t-s
    end
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

function InchDiag_MC(t, dt, H0, W, O, Bath, M, psi0, Nwlk)
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
            G_par = Compute_Inchworm(ti, tf, dt, t, H0, W, O, G, Bath, psi0, M, Nwlk)
            push!(G, G_par)
        end
    end
    return last(G)
end

function Compute_Inchworm(ti, tf, dt, t, H0, W, O, G, Bath, psi0, M, Nwlk)
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
        return MC_diag(ti, dt, t, H0, W, O, Bath, M, psi0, Nwlk)
    else
        # This is the only case were new functions have to be implemented as Inchworm algorithm is implemented in
        # this part
        return Inchworm_expand(ti, tf, dt, t, H0, W, O, G, Bath, M, psi0, Nwlk)
    end
end

function Inchworm_expand(ti, tf, dt, t, H0, W, O, G, Bath, M, psi0, Nwlk)
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
    G_new = Compute_bold(ti, tf, ti, tf, tf-dt, t, H0, O, G)

    # Now diagrammatic expansion has to be considered
    for i in 1:M
        G_new = G_new + InchM_sample(ti, tf, dt, t, H0, W, O, G, Bath, psi0, i, Nwlk)
    end
    return G_new
end

function InchM_sample(ti, tf, dt, t, H0, W, O, G, Bath, psi0, M, Nwlk)
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
    Norm_fact =0
    Xrand = zeros(2*M)
    Xrand[1] = tf-dt+dt*rand(Float64)
    for i in 2:2*M
        Xrand[i]= ti+(tf-ti)*rand(Float64)
    end
    Gpar = Compute_Inchpropagator(ti, tf, dt, t, Xrand, H0, W, O, G, Bath)
    Pi = abs(tr(Gpar*psi0))
    for i in 1:Nwlk
        # Here 2M random points on the interval ti-tf have to be sampled
        # For the Inchworm procedure, the sampling process is a bit more complex as the sampling must ensure
        # Inchworm properness so at least the first point has to be sampled within the interval [tf-dt,tf]
        Gpar, P = Metropolis_Hasting_Inch(ti, tf, dt, t, H0, W, O, Bath, Xrand, Pi, psi0, G)
        G_stoch += Gpar
        Norm_fact += P
        #Xrand = zeros(2*M)
        #Xrand[1] = tf-dt+dt*rand(Float64)
        #for i in 2:2*M
        #    Xrand[i] = ti+(tf-ti)*rand(Float64)
        #end
        #Xrand = sort(Xrand)
        #G_stoch += Compute_Inchpropagator(ti,tf,dt,t,Xrand,H0,W,O,G,Bath)
    end
    # besides the average the integral value has to be corrected by the volume of the hyperspace of integration
    # CAREFUL perhaps this hypervolume has to be modified since the sampling method has changed
    # Again the volume has to be computed for two different cases
    if ti<t && t<tf
        V=0
        for i in 1:2*M
            V+= dt*(tf-t)^(i-1)/fac(i-1)*(t-ti)^(2*M-i)/fac(2*M-i)
        end
    else
        V = dt*(tf-ti)^(2*M-1)/fac(2*M-1)
    end
    Norm_fact/=Nwlk
    return V*G_stoch/(Nwlk*Norm_fact)
end

function Metropolis_Hasting_Inch(ti, tf, dt, t, H0, W, O, Bath, Xrand, Pi, psi0, G)
    """
    Metropolis_Hasting_Inch comptes the Metropolis samplign for the Inchworm algorithm
    ti,tf,t,dt: time parameters
    H0: Unperturbed Hamiltonian
    W: perturbation
    O: Observable
    Bath: Bath parameters
    Xrand: Initial sampled point
    Pi: Initial probability
    psi0: Initial density matrix
    """
    Pold=Pi
    Pnew = 0
    L = size(Xrand)[1]
    Xrandold=Xrand
    Xrandnew = zeros(L)

    # First an equilibration part is computed
    for i in 1:20
        # displacement is build taking into account that the interval must remain Inchworm proper

        drand = 10*dt.*(0.5.-rand(Float64,L))
        drand[L]=(dt/5).*(0.5.-rand(Float64))
        Xrand1 = zeros(L)
        Xrand1= ti.+(mod.((Xrandold+drand).-ti,tf-ti))
        Xrand1[L] = tf-dt+mod(Xrandold[L]+drand[L]-tf-dt,dt)
        coin_flip = rand(Float64)
        Gnew = Compute_Inchpropagator(ti,tf,dt,t,Xrand1,H0,W,O,G,Bath)
        Pnew = abs(tr(Gnew*psi0))
        if Pnew/Pold>1
            Xrandolg=Xrand1
            Pold=Pnew
        elseif coin_flip<=Pnew/Pold
            Xrandold = Xrand1
            Pold=Pnew            
        end
    end

    Lop = size(H0)[1]
    Gpar = zeros(Lop, Lop)
    Ppar = 0

    # Now Equilibration is computed 
    for i in 1:30
        drand = 10*dt.*(0.5.-rand(Float64,L))
        drand[L]=(dt/5).*(0.5.-rand(Float64))
        Xrand1 = zeros(L)
        Xrand1= ti.+(mod.((Xrandold+drand).-ti,tf-ti))
        Xrand1[L] = tf-dt+mod(Xrandold[L]+drand[L]-tf-dt,dt)
        coin_flip = rand(Float64)
        Gnew = Compute_Inchpropagator(ti,tf,dt,t,Xrand1,H0,W,O,G,Bath)
        Pnew = abs(tr(Gnew*psi0))
        if Pnew/Pold>1
            Xrandolg=Xrand1
            Pold=Pnew
        elseif coin_flip<=Pnew/Pold
            Xrandold = Xrand1
            Pold=Pnew            
        end
        Gpar += Compute_Inchpropagator(ti,tf,dt,t,Xrandold,H0,W,O,G,Bath)/Pold
        Ppar += 1/Pold
    end

    Gpar /=30
    Ppar /=30

    return Gpar, Ppar
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
    LBath = Compute_bathInch(Xrand, tf, dt, Bath, t)
    # First part of the propagations is done separately
    # Now, as here I'm resummating, the Green function is no longer made up of bare propagators but rather on bold
    # ones
    Gp = Compute_bold(ti, tf, ti, Xrand[1], tf-dt, t, H0, O, G)
    Gp = W * Gp
    for i in 2:L
        Gp = Compute_bold(ti, tf, Xrand[i-1], Xrand[i], tf-dt, t, H0, O, G) * Gp
        Gp = W * Gp
    end
    # Last part of the propagations is done separately
    Gp = Compute_bold(ti, tf, last(Xrand), tf, tf-dt, t, H0, O, G) * Gp
    # Now all boring prefactors have to be calculated
    prefact = 1im^L
    for i in 1:L
        if Xrand[i]<t
            prefact *=-1
        end
    end
    return prefact*Gp*LBath
end

function Compute_bathInch(Xrand, tf, dt, Bath, t)
    """
    Compute_bathInch computes the sum of possible factorizations that are Inchworm proper
    Xrand: sample of points
    tf,dt: time variables
    Bath: bath properties
    t: physical time
    """
    N = size(Xrand)[1]
    Partition = Pairwise_partitions(N)
    L = size(Partition)[1]
    Bath_influence = 0
    for i in 1:L
        if Is_Inch(Partition[i], Xrand, dt, tf)
            Bath_influence += Compute_product(Xrand, Bath, Partition[i], t)
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

function main(dt, t, H0, W, O, psi0, Bath, M, Nwlk)
    """
    main functions just assembles everything
    dt: time step
    t: total time propagation
    H0: Unperturbed Hamiltonian
    W: Perturbation
    O: observable
    psi0: initial wavefunction (in matrix form)
    Bath: bath parameters
    M: order of diagrammatic expansion
    Nwlk: Number or walkers
    """
    Nsims = floor(Int,t/dt)
    G = []
    for i in 1:Nsims
        push!(G, InchDiag_MC(i*dt,dt,H0,W,O,Bath,M,psi0,Nwlk))
        println(i)
    end
    Obs= []
    L = size(G)[1]
    for i in 1:L
        push!(Obs, real(tr(psi0*G[i])))
        println(real(tr(psi0*G[i])),"\t",i*dt)
        #println(G[i])
    end
    return Obs
end

t=4
dt = 1/8
H0 = zeros(2,2)
H0[1,2] = 1
H0[2,1] = 1
W = zeros(2,2)
W[1,1] = 1
W[2,2] = -1
O = zeros(2,2)
O[1,1]=1
O[2,2]=-1
M=1
psi0 = zeros(2,2)
psi0[1,1] = 1
Nwlk = 100
Bath=[0.1,5,0.5]
L=main(dt, t, H0, W, O, psi0, Bath, M, Nwlk)
filename="B.txt"
writedlm(filename,L)
