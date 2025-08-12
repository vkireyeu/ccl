#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <stdio.h>
#include <stdexcept>

#include "CCParticle.h"
#include "SkyrmeFormula.h"
#include "EnergyUtils.h"


class CClusterizer{
    public:
        enum class SAStepsMode {Fixed, Linear, Exponential}; 
        enum class SAProbMode  {Fixed, Linear}; 
    
    private:
        double Radius            = 0;
        double Momentum          = 0;
        bool   CheckMomentum     = false;
        bool   CheckPairMomentum = false;
        bool   ComputeEbind      = false;
        bool   TrueTimeMST       = false;
        std::vector<CCParticle> particles;

        // No default Skyrme
        SkyrmeFormula* skyrmeFormula = nullptr;

        // Coalescence procedure settings
        bool   cls_UseEbind      = false; // Switch for the usage of E_bind criterion instead of probabilities
        bool   cls_UseIsoEbind   = false; // Switch for the usage of E_bind criterion WITH probabilities

        // Simulated Annealing procedure settings
        double sa_Tmax        =    1.0;    // Initial "temperature"
        double sa_Tmin        =   1e-5;    // Minimal tempersture
        double sa_Cool        =   0.95;    // Cooling rate
        int    sa_Steps       =    500;    // Accepted/rejected trials per T (initial!)
        double sa_Pnew        =   0.25;    // Probability to put a moved baryon into a new cluster
        double sa_PnewMin     =   0.25;    // Final probability for the case of linear decrease
        bool   sa_MHCriterion =  false;    // Switch for the usage of Metropolis–Hastings criterion
        double sa_EbindBound  = -0.004;    // Ebind/A for 'bound' clusters selection within makeSAchain()
  
        SAStepsMode sa_StepsMode = SAStepsMode::Exponential;
        SAProbMode  sa_PnewMode  = SAProbMode::Fixed;


        // Tracking settings
        bool   tr_UseEbind    = true;      // Use binding energy as cluster stability criteria during tracking
        double tr_EbindCut    = 0.0;       // Ebind cut (Ebind/A) in GeV


        // Random engine
        static std::mt19937& get_rng() {
            thread_local std::random_device rd;
            thread_local std::mt19937 eng(rd());
            return eng;
        }
        static std::uniform_real_distribution<double>& U01() {
            thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
            return dist;
        }

    public:
        CClusterizer();
        virtual ~CClusterizer();


        void setMomentum(double momentum)     {Momentum      = momentum;}
        void setCheckPairMomentum(bool check) {CheckMomentum =    check;}
        void setCheckMomentum(bool check)     {CheckMomentum =    check;}
        void setComputeEbind(bool check)      {ComputeEbind  =    check;}
        void setTrueTimeMST(bool check)       {TrueTimeMST   =    check;}

        void setRadius(double radius) {
            if (radius < 0.0) {
                throw std::invalid_argument("Radius cannot be negative");
            }
            Radius = radius;
        }

        void setMomentumWithCheck(double momentum, bool check) {
            Momentum = momentum;
            CheckMomentum = check;
        }

        void setSkyrmeFormula(const SkyrmeFormula& f) {
            if (skyrmeFormula) delete skyrmeFormula;
            skyrmeFormula = new SkyrmeFormula(f);
        }

        double getRadius()            const {return Radius;}
        double getMomentum()          const {return Momentum;}
        bool   getCheckPairMomentum() const {return CheckPairMomentum;}
        bool   getCheckMomentum()     const {return CheckMomentum;}
        bool   getComputeEbind()      const {return ComputeEbind;}
        bool   getTrueTimeMST()       const {return TrueTimeMST;}

        bool   hasSkyrmeFormula()     const {return skyrmeFormula != nullptr;}

        const  SkyrmeFormula& getSkyrmeFormula() const {
            if (!skyrmeFormula) {
                throw std::runtime_error("Skyrme formula not set");
            }
            return *skyrmeFormula;
        }


        CCParticle mergeToCluster(const std::vector<CCParticle*>& group) const;

        bool shouldCoalesce(CCParticle& a, CCParticle& b) const; 
        void makeCoalescence(std::vector<CCParticle>& particles,
                                           ParticleType clusterType,
                                           double radius,
                                           double maxMomentum,
                                           bool checkMomentum,
                                           float probability);

        bool proximityCheckMST(const CCParticle& p1, const CCParticle& p2) const;
        bool passesMomentumCheck(const std::vector<CCParticle*>& group) const;
        std::vector<CCParticle> makeMST(const std::vector<CCParticle>& particles);


        // Binding energy
        static double Ec2;   // GeV * fm
        static double Al;    // fm^2
        static double Sal;   // fm^2
        static double Alpha; // GeV 
        static double Beta;  // GeV 
        static double Gamma;
        static double HyPot; // Fraction of the hyperon potential vs normal particles
        static bool   ComputeSkyrme;
        static bool   ComputeCoulomb;
        static bool   ComputeEasy;

        static void setEc2(double value)   {Ec2   = value;}
        static void setAl(double value)    {Al    = value;}
        static void setSal(double value)   {Sal   = value;}
        static void setAlpha(double value) {Alpha = value;}
        static void setBeta(double value ) {Beta  = value;}
        static void setGamma(double value) {Gamma = value;}
        static void setHyPot(double value) {HyPot = value;}
        static void setAlphaBetaGamma(double A, double B, double G){
            Alpha = A; Beta  = B; Gamma = G;
        }
        static void setComputeSkyrme(bool check) {ComputeSkyrme  = check;}
        static void setComputeCoul(bool check)   {ComputeCoulomb = check;}
        static void setComputeEasy(bool check)   {ComputeEasy    = check;}

        static double getEc2()           {return Ec2;}
        static double getAl()            {return Al;}
        static double getSal()           {return Sal;}
        static double getAlpha()         {return Alpha;}
        static double getBeta()          {return Beta;}
        static double getGamma()         {return Gamma;}
        static double getHyPot()         {return HyPot;}
        static double getAlpha0()        {return 0.5 * Alpha / Sal;}
        static double getBeta0()         {return Beta / ((Gamma + 1.0) * std::pow(Sal, Gamma));}
        static bool   getComputeSkyrme() {return ComputeSkyrme;}
        static bool   getComputeCoul()   {return ComputeCoulomb;}
        static bool   getComputeEasy()   {return ComputeEasy;}



        // Coalescence
        void setClsUseEbind(bool val) {
            cls_UseEbind = val;
            if (val) cls_UseIsoEbind = false;
        }
        void setClsUseIsoEbind(bool val) {
            cls_UseIsoEbind = val;
            if (val) cls_UseEbind = false;
        }

        bool getClsUseEbind()    const   {return cls_UseEbind;}
        bool getClsUseIsoEbind() const   {return cls_UseIsoEbind;}


        // Simulated Annealing
        void setSATmax(double val)       {sa_Tmax        = val;}
        void setSATmin(double val)       {sa_Tmin        = val;}
        void setSACool(double val)       {sa_Cool        = val;}
        void setSASteps(double val)      {sa_Steps       = val;}
        void setSAPnew(double val)       {sa_Pnew        = val;}
        void setSAPnewMin(double val)    {sa_PnewMin     = val;}
        void setMHCriterion(bool val)    {sa_MHCriterion = val;}
        void setSAEbindBound(double val) {sa_EbindBound  = val;}

        double getSATmax()       const {return sa_Tmax;}
        double getSATmin()       const {return sa_Tmin;}
        double getSACool()       const {return sa_Cool;}
        int    getSASteps()      const {return sa_Steps;}
        double getSAPnewMin()    const {return sa_PnewMin;}
        bool   getMHCriterion()  const {return sa_MHCriterion;}
        double getSAEbindBound() const {return sa_EbindBound;}

        void setSAStepsMode(SAStepsMode val) {sa_StepsMode = val;}
        void setSAPnewMode(SAProbMode val)   {sa_PnewMode  = val;}

        SAStepsMode getSAStepsMode() const {return sa_StepsMode;}
        SAProbMode  getSAPnewMode()  const {return sa_PnewMode;}

        // Classic simulated annealing implementation
        std::vector<CCParticle> makeSA(const std::vector<CCParticle>& particles);
        // Growth-only simulated annealing implementation
        std::vector<CCParticle> makeSA2(const std::vector<CCParticle>& particles);
        // Simulated annealing chain: makeSA() -> makeSA2()
        std::vector<CCParticle> makeSAchain(const std::vector<CCParticle>& particles);


        // Stability tracking 
        void updateStableVector(const std::vector<CCParticle>& input, const float step_time,
                                std::vector<CCParticle>& output);

        void setTrackingUseEbind(bool val)   {tr_UseEbind = val;}
        void setTrackingEbindCut(double val) {tr_EbindCut = val;}

        bool getTrackingUseEbind()   const {return tr_UseEbind;}
        double getTrackingEbindCut() const {return tr_EbindCut;}
};

