import {Component, Inject, OnInit, ViewChild} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {ModelStartData} from "../models";
import {ModelService} from "./model.service";
import {Helpers} from "../helpers";

@Component({
    selector: 'model-start-dialog',
    templateUrl: 'model-start-dialog.html',
    styleUrls: ['./model-start-dialog.css']
})
export class ModelStartDialogComponent implements OnInit {
    error: string;
    @ViewChild('textarea') textarea;

    constructor(
        public dialogRef: MatDialogRef<ModelStartDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: ModelStartData,
        protected service: ModelService
    ) {

    }

    on_ok_click(): void {
        if (this.data && this.data.pipe && this.data.pipe.versions) {
            let version = this.data.pipe.versions[0];
            if (version.name == 'last') {
                version.name = Helpers.format_date_time(new Date(),
                    true, true);
            }
        }

        this.service.start_end(this.data).subscribe(res => {
            this.error = res.error;
            if (res.success) {
                this.dialogRef.close();
            }
        });
    }

    on_cancel_click(): void {
        this.dialogRef.close();
    }

    dag_changed() {
        let dag = this.data.dag;
        if (!dag) {
            return;
        }
        for (let pipe of dag.pipes) {
            if (!pipe.versions ||
                pipe.versions.length == 0) {
                pipe.versions = [
                    {'name': 'last', 'equations': ''}
                ];
            }

        }
        if (dag.pipes.length >= 1) {
            this.data.pipe = dag.pipes[0];
            this.data.pipe.version = this.data.pipe.versions[0];
        }
    }

    pipe_changed(){
        let pipe = this.data.pipe;
        if(!pipe || !pipe.versions || pipe.versions.length == 0){
            return;
        }

        pipe.version = pipe.versions[0];
    }

    ngOnInit(): void {
        let self = this;
        this.service.start_begin(this.data).subscribe(res => {
            this.error = res.error;
            if (this.error) {
                return;
            }
            this.data = {
                'model_id': self.data.model_id,
                'dags': res.dags
            } as ModelStartData;

            if (res.dag) {
                for (let d of this.data.dags) {
                    if (d.id == res.dag.id) {
                        this.data.dag = d;
                        break
                    }
                }
            }

            this.dag_changed();
        })
    }

    key_down(event) {
        if (!this.data.pipe || !this.data.pipe.version) {
            return;
        }
        if(event.ctrlKey && event.key != 'v'){
            return;
        }
        let version = this.data.pipe.version;
        if (version.name != 'last') {
            let new_version = {
                'name': 'last',
                'equations': event.target.value
            };
            this.data.pipe.version = new_version;
            if (this.data.pipe.versions[0].name == 'last') {
                this.data.pipe.versions[0] = new_version;
            } else {
                this.data.pipe.versions.splice(0, 0,
                    new_version);
            }

        }

        let content = Helpers.handle_textarea_down_key(event,
            this.textarea.nativeElement);
        if (content) {
            this.data.pipe.version.equations = content;
        }


    }

    move_up(event) {
        event.preventDefault();
        event.stopPropagation();

        if (!this.data.pipe || !this.data.pipe.version) {
            return;
        }
        let index = this.data.pipe.versions.indexOf(this.data.pipe.version);
        if (index == 0) {
            return;
        }

        this.data.pipe.version = this.data.pipe.versions[index - 1];
    }

    move_down(event) {
        event.preventDefault();
        event.stopPropagation();

        if (!this.data.pipe || !this.data.pipe.version) {
            return;
        }
        let index = this.data.pipe.versions.indexOf(this.data.pipe.version);
        if (index >= this.data.pipe.versions.length - 1) {
            return;
        }

        this.data.pipe.version = this.data.pipe.versions[index + 1];
    }

    is_up_transparent() {
        if (!this.data.pipe || !this.data.pipe.version) {
            return true;
        }
        let index = this.data.pipe.versions.indexOf(this.data.pipe.version);
        return index <= 0;

    }

    is_down_transparent() {
        if (!this.data.pipe || !this.data.pipe.version) {
            return true;
        }
        let index = this.data.pipe.versions.indexOf(this.data.pipe.version);
        return index >= this.data.pipe.versions.length - 1;

    }
}