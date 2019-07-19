import {AfterViewInit, Component} from '@angular/core';
import {MessageService} from '../../../message.service'
import {DagDetailService} from "../dag-detail/dag-detail.service";
import {DynamicresourceService} from "../../../dynamicresource.service";

@Component({
    selector: 'app-config',
    templateUrl: './config.component.html',
    styleUrls: ['./config.component.css']
})
export class ConfigComponent implements AfterViewInit {
    public dag: number;
    config: string;

    constructor(private message_service: MessageService,
                private service: DagDetailService,
                private resource_service: DynamicresourceService,
    ) {
    }

    ngAfterViewInit() {
        let self = this;
        this.resource_service.load(
            'prettify',
            'prettify-yaml',
            'prettify-css').then(() => {
            self.service.get_config(self.dag).subscribe(res => {
                let node = document.createElement('pre');
                node.textContent = res.data;
                node.className = "prettyprint linenums lang-yaml";
                let codeholder = document.getElementById(
                    'codeholder');

                codeholder.appendChild(node);
                window['PR'].prettyPrint();
            });
        })
    }

}
