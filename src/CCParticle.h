#pragma once
#include <cmath>
#include <vector>
#include <unordered_map>


enum class ParticleType : short {
    Unset        =  -2,
    Unknown      =  -1,
    Neutron      =   0,
    Proton       =   1,
    LambdaSigma  =   2,
    Deuteron     =   3,
    Triton       =   4,
    Helium3      =   5,
    Alpha        =   6,
    H3L          =   7,
    H4L          =   8,
    MSTNucleus   = 100
};


class CCParticle {
    private:
        short  ClusterID     = -1;
        long   PID           =  0;
        short  Number        =  0;
        short  Charge        =  0;
        double Time          = 0.;
        double CollisionTime = 0.;
        double X             = 0.;
        double Y             = 0.;
        double Z             = 0.;
        double Px            = 0.;
        double Py            = 0.;
        double Pz            = 0.;
        double Mass          = 0.;
        bool   InCluster     = false;
        ParticleType Type    = ParticleType::Unset;
        double BindingEnergy = 0.0;
        int  UniqueID        =  0;
        std::vector<CCParticle> Childs;

    public:
        CCParticle();
        virtual ~CCParticle();

        // Setters
        void setClusterID(short id)     {ClusterID = id;}
        void setNumber(short nr)        {Number = nr;}
        void setCharge(short charge)    {Charge = charge;}
        void setCollisionTime(double t) {CollisionTime = t;}
        void setTime(double time)       {Time = time;}
        void setX(double x)             {X = x;}
        void setY(double y)             {Y = y;}
        void setZ(double z)             {Z = z;}
        void setPx(double px)           {Px = px;}
        void setPy(double py)           {Py = py;}
        void setPz(double pz)           {Pz = pz;}
        void setMass(double mass)       {Mass = mass;}
        void setInCluster(bool flag)    {InCluster = flag;}
        void setUniqueID(int uid)       {UniqueID = uid;}
        void addChild(const CCParticle* p) {
            if (p) Childs.push_back(*p);
            else   Childs.emplace_back();
        }
        void addChild(const CCParticle p) {
            Childs.push_back(p);
        }

        void setPID(long pid) {
            PID = pid;
            Type = classifyPID(pid);
        }

        // Combined setter for IDs
        void setIDS(short cls, long pid, short nr) {
            ClusterID =  cls;
            Number    =   nr;
            setPID(pid);
        }

        // Combined setter for position coordinates
        void setPosition(double x, double y, double z, double t) {
            X    = x;
            Y    = y;
            Z    = z;
            Time = t;
        }

        // Combined setter for momentum components
        void setMomentum(double px, double py, double pz, double m) {
            Px   = px;
            Py   = py;
            Pz   = pz;
            Mass =  m;
        }


        // Getters
        short  getClusterID()     const {return ClusterID;}
        long   getPID()           const {return PID;}
        short  getNumber()        const {return Number;}
        short  getCharge()        const {return Charge;}
        double getTime()          const {return Time;}
        double getCollisionTime() const {return CollisionTime;}
        double getX()             const {return X;}
        double getY()             const {return Y;}
        double getZ()             const {return Z;}
        double getPx()            const {return Px;}
        double getPy()            const {return Py;}
        double getPz()            const {return Pz;}
        double getMass()          const {return Mass;}
        bool   getInCluster()     const {return InCluster;}
        ParticleType getType()    const {return Type;}
        double getP()             const {return std::sqrt(Px*Px + Py*Py + Pz*Pz);}
        double getPt()            const {return std::hypot(Px, Py);}
        double getE()             const {return std::sqrt(Px*Px + Py*Py + Pz*Pz + Mass*Mass);}
        int   getUniqueID()       const {return UniqueID;}
        bool  hasChilds()         const {return !Childs.empty();}

        const std::vector<CCParticle>& getChilds() const {return Childs;}

        // Needed for operatons over pointers like the binding energy calulation etc
        std::vector<CCParticle*> getChildsPtrs(){
            std::vector<CCParticle*> v;
            v.reserve(Childs.size());
            for (auto &x : Childs) v.push_back(&x);
            return v;
        }

        double getRapidity() const {
            const double E  = this -> getE();
            return 0.5 * std::log((E + Pz) / (E - Pz));
        }
        double getBx()	     const {return Px / this -> getE();}
        double getBy()	     const {return Py / this -> getE();}
        double getBz()	     const {return Pz / this -> getE();}


        void LorentzBoost(double bx, double by, double bz);

        ParticleType classifyPID(long pid);

        void setBindingEnergy(double e) {BindingEnergy = e;}
        double getBindingEnergy() const {return BindingEnergy;}


        void clearChilds() {Childs.clear();}
};
