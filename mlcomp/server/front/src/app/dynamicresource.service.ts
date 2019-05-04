import {Injectable} from '@angular/core';

interface Resource {
    name: string;
    src: string;
    type: string
}

export const ResourceStore: Resource[] = [
    {name: 'vis.min.js', src: 'assets/visjs/vis.min.js', type: 'js'},
    {name: 'vis.min.css', src: 'assets/visjs/vis.min.css', type: 'css'},
    {name: 'prettify', src: 'assets/prettify/prettify.js', type: 'js'},
    {name: 'prettify-yaml', src: 'assets/prettify/lang-yaml.js', type: 'js'},
    {name: 'prettify-css', src: 'assets/prettify/prettify.css', type: 'css'}
];


declare var document: any;

@Injectable({providedIn: 'root'})
export class DynamicresourceService {

    private resources: any = {};

    constructor() {
        ResourceStore.forEach((resource: any) => {
            this.resources[resource.name] = {
                loaded: false,
                src: resource.src,
                type: resource.type
            };
        });
    }

    load(...resources: string[]) {
        const promises: any[] = [];
        resources.forEach((resource) => promises.push(this.loadResource(resource)));
        return Promise.all(promises);
    }

    loadResource(name: string) {
        return new Promise((resolve, reject) => {
            if (!this.resources[name].loaded) {
                //load resource
                let resource = document.createElement(this.resources[name].type == 'js' ? 'script' : 'link');
                if (this.resources[name].type == 'js') {
                    resource.type = 'text/javascript';
                    resource.async = true;
                    resource.src = this.resources[name].src;
                } else {
                    resource.type = 'text/css';
                    resource.rel = 'stylesheet';
                    resource.href = this.resources[name].src;
                }

                if (resource.readyState) {  //IE
                    resource.onreadystatechange = () => {
                        if (resource.readyState === "loaded" || resource.readyState === "complete") {
                            resource.onreadystatechange = null;
                            this.resources[name].loaded = true;
                            resolve({resource: name, loaded: true, status: 'Loaded'});
                        }
                    };
                } else {  //Others
                    resource.onload = () => {
                        this.resources[name].loaded = true;
                        resolve({resource: name, loaded: true, status: 'Loaded'});
                    };
                }
                resource.onerror = (error: any) => resolve({resource: name, loaded: false, status: 'Loaded'});
                document.getElementsByTagName('head')[0].appendChild(resource);
            } else {
                resolve({resource: name, loaded: true, status: 'Already Loaded'});
            }
        });
    }

}
