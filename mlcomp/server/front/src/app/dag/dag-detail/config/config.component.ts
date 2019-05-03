import {Component, OnInit} from '@angular/core';
import {MessageService} from '../../../message.service'
import {ActivatedRoute} from "@angular/router";
import {DagDetailService} from "../../../dag-detail.service";

@Component({
    selector: 'app-config',
    templateUrl: './config.component.html',
    styleUrls: ['./config.component.css']
})
export class ConfigComponent implements OnInit {
    private dag_id: string;
    config: string;

    constructor(private message_service: MessageService, private route: ActivatedRoute,
                private service: DagDetailService
    ) {
    }

    ngOnInit() {
        this.load_prettify();
        this.dag_id = this.route.parent.snapshot.paramMap.get('id');

    }

    load_prettify() {
        let self = this;
        this.message_service.add('loading prettify');
        let scripts_to_load = ['assets/prettify/prettify.js', 'assets/prettify/lang-yaml.js'];
        for (let s of scripts_to_load) {
            let node = document.createElement('script');
            node.src = s;
            node.type = 'text/javascript';
            node.async = true;

            if (s == 'assets/prettify/prettify.js') {
                node.onload = function () {
                    self.service.get_config(self.dag_id).subscribe(res => {
                        let node = document.createElement('pre');
                        node.textContent = res.data;
                        node.className = "prettyprint linenums lang-yaml";
                        document.getElementById('codeholder').appendChild(node);
                        window['PR'].prettyPrint();

                    });
                }
            }

            document.getElementsByTagName('head')[0].appendChild(node);
        }


        let node2 = document.createElement('link');
        node2.href = "assets/prettify/prettify.css";
        node2.type = 'text/css';
        node2.rel = 'stylesheet';
        document.getElementsByTagName('head')[0].appendChild(node2);
    }

}
