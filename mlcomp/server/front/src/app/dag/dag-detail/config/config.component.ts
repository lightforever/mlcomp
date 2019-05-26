import {Component, OnInit} from '@angular/core';
import {MessageService} from '../../../message.service'
import {ActivatedRoute} from "@angular/router";
import {DagDetailService} from "../../../dag-detail.service";
import {DynamicresourceService} from "../../../dynamicresource.service";

@Component({
    selector: 'app-config',
    templateUrl: './config.component.html',
    styleUrls: ['./config.component.css']
})
export class ConfigComponent implements OnInit {
    private dag_id: number;
    config: string;

    constructor(private message_service: MessageService, private route: ActivatedRoute,
                private service: DagDetailService,
                private resource_service: DynamicresourceService,
    ) {
    }

    ngOnInit() {
        let self = this;
        this.dag_id = parseInt(this.route.parent.snapshot.paramMap.get('id'));
        this.resource_service.load('prettify', 'prettify-yaml', 'prettify-css').then(() => {
            self.service.get_config(self.dag_id).subscribe(res => {
                let node = document.createElement('pre');
                node.textContent = res.data;
                node.className = "prettyprint linenums lang-yaml";
                document.getElementById('codeholder').appendChild(node);
                window['PR'].prettyPrint();
            });
        })
    }

}
