
/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, int period_offset, long int offset) {
        this->ranks = ranks;
        this->period_ = period;
        this->period_offset_ = period_offset;
        this->offset_ = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    virtual void record() = 0;
    virtual void record_targets() = 0;
    virtual long int size_in_bytes() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period_;
    int period_offset_;
    long int offset_;

};

class PopRecorder0 : public Monitor
{
public:
    PopRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset) {

        this->rates = std::vector< std::vector< double > >();
        this->record_rates = false; 
        this->p = std::vector< std::vector< double > >();
        this->record_p = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop0.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; 
    }

    void record() {

        if(this->record_rates && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->rates.push_back(pop0.rates); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.rates[this->ranks[i]]);
                }
                this->rates.push_back(tmp);
            }
        }
        if(this->record_p && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->p.push_back(pop0.p); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.p[this->ranks[i]]);
                }
                this->p.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop0.r); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop0.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop0.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop0.spiked[i])!=this->ranks.end() ){
                        this->spike[pop0.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        size_in_bytes += sizeof(double) * rates.capacity();	//rates
        size_in_bytes += sizeof(double) * p.capacity();	//p
        size_in_bytes += sizeof(double) * r.capacity();	//r
        
        return size_in_bytes;
    }


    // Local variable rates
    std::vector< std::vector< double > > rates ;
    bool record_rates ; 
    // Local variable p
    std::vector< std::vector< double > > p ;
    bool record_p ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
        }
    }

};

class PopRecorder1 : public Monitor
{
public:
    PopRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset) {

        this->g_exc = std::vector< std::vector< double > >();
        this->record_g_exc = false; 
        this->tau_m = std::vector< std::vector< double > >();
        this->record_tau_m = false; 
        this->tau_e = std::vector< std::vector< double > >();
        this->record_tau_e = false; 
        this->vt = std::vector< std::vector< double > >();
        this->record_vt = false; 
        this->vr = std::vector< std::vector< double > >();
        this->record_vr = false; 
        this->El = std::vector< std::vector< double > >();
        this->record_El = false; 
        this->Ee = std::vector< std::vector< double > >();
        this->record_Ee = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop1.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; 
    }

    void record() {

        if(this->record_tau_m && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->tau_m.push_back(pop1.tau_m); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.tau_m[this->ranks[i]]);
                }
                this->tau_m.push_back(tmp);
            }
        }
        if(this->record_tau_e && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->tau_e.push_back(pop1.tau_e); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.tau_e[this->ranks[i]]);
                }
                this->tau_e.push_back(tmp);
            }
        }
        if(this->record_vt && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->vt.push_back(pop1.vt); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.vt[this->ranks[i]]);
                }
                this->vt.push_back(tmp);
            }
        }
        if(this->record_vr && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->vr.push_back(pop1.vr); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.vr[this->ranks[i]]);
                }
                this->vr.push_back(tmp);
            }
        }
        if(this->record_El && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->El.push_back(pop1.El); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.El[this->ranks[i]]);
                }
                this->El.push_back(tmp);
            }
        }
        if(this->record_Ee && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->Ee.push_back(pop1.Ee); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.Ee[this->ranks[i]]);
                }
                this->Ee.push_back(tmp);
            }
        }
        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->v.push_back(pop1.v); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop1.r); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
        if(this->record_spike){
            for(int i=0; i<pop1.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop1.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop1.spiked[i])!=this->ranks.end() ){
                        this->spike[pop1.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_exc && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->g_exc.push_back(pop1.g_exc); 
            else{
                std::vector<double> tmp = std::vector<double>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.g_exc[this->ranks[i]]);
                }
                this->g_exc.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        size_in_bytes += sizeof(double) * tau_m.capacity();	//tau_m
        size_in_bytes += sizeof(double) * tau_e.capacity();	//tau_e
        size_in_bytes += sizeof(double) * vt.capacity();	//vt
        size_in_bytes += sizeof(double) * vr.capacity();	//vr
        size_in_bytes += sizeof(double) * El.capacity();	//El
        size_in_bytes += sizeof(double) * Ee.capacity();	//Ee
        size_in_bytes += sizeof(double) * v.capacity();	//v
        size_in_bytes += sizeof(double) * r.capacity();	//r
        
        return size_in_bytes;
    }


    // Local variable g_exc
    std::vector< std::vector< double > > g_exc ;
    bool record_g_exc ; 
    // Local variable tau_m
    std::vector< std::vector< double > > tau_m ;
    bool record_tau_m ; 
    // Local variable tau_e
    std::vector< std::vector< double > > tau_e ;
    bool record_tau_e ; 
    // Local variable vt
    std::vector< std::vector< double > > vt ;
    bool record_vt ; 
    // Local variable vr
    std::vector< std::vector< double > > vr ;
    bool record_vr ; 
    // Local variable El
    std::vector< std::vector< double > > El ;
    bool record_El ; 
    // Local variable Ee
    std::vector< std::vector< double > > Ee ;
    bool record_Ee ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
        }
    }

};

class ProjRecorder0 : public Monitor
{
public:
    ProjRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;

    };
    void record() {

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(proj0.w[this->ranks[i]]);
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };
    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

