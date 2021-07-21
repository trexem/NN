#ifndef _NEURON_H_
#define _NEURON_H_

#define RELU 1
#define SIGM 2
#define SFMX 3
#define PRELU 4

#include <iostream>
#include <math.h>
using namespace std;

class Neuron {
private:
	double m_val{0};
	double m_derived_val{0};
	double m_active_val{0};
	int m_activation_type{1};
public:
//Constructors
	Neuron(double t_val){
		m_val = t_val;
	}
	Neuron(double t_val, int t_activation_type){
		m_val = t_val;
		m_activation_type = t_activation_type;
	}
//Functions
	void activate(){
		if(m_activation_type == RELU) {
			if(m_val > 0) {
				m_active_val = m_val;
			} else{
				m_active_val = 0;
			}
		} else if(m_activation_type == PRELU) {
			if(m_val>0) {
				m_active_val = m_val*5;
			} else{
				m_active_val = m_val*.001;
			}
		} else if(m_activation_type==SIGM) {
			m_active_val = (1 / (1+exp(-m_val)));
		} else if(m_activation_type = SFMX) {
			m_active_val = m_val;
		} else{
			m_active_val = (1 / (1+exp(-m_val)));
		}
	}
	void derive(){
		if (m_activation_type == RELU) {
			if (m_val>0) {
				m_derived_val = 1;
			} else {
				m_derived_val = 0;
			}
		} else if (m_activation_type == PRELU) {
			if (m_val > 0) {
				m_derived_val = .5;
			} else {
				m_derived_val = .001;
			}
		} else if(m_activation_type == SIGM) {
			m_derived_val = m_active_val*(1-m_active_val);
		} else if (m_activation_type == SFMX) {
			m_derived_val = m_active_val;
		} else{
			m_derived_val = m_active_val*(1-m_active_val);
		}
	}

//setter
	void setVal(double t_val){
		m_val=t_val;
		activate();
		derive();
	}

//Getters
	double getVal(){
		return m_val;
	}
	double getDerivedVal(){
		return m_derived_val
	};
	double getActiveVal(){
		return m_active_val
	};
};
#endif
